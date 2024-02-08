from pathlib import Path
import os
from models import EEGVitPretrained
import lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import splitfolders
from utils import set_global_seed, mean_std
import logging
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar
import shutil
from calflops import calculate_flops
local_home = Path(__file__).parent.resolve()
os.environ["HF_HUB_CACHE"] = f"{local_home}/tmp"
os.environ["HF_HOME"] = f"{local_home}/tmp"
os.environ['TRANSFORMERS_CACHE'] = f"{local_home}/tmp"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['NUMBA_CACHE_DIR'] = f"{local_home}/tmp"
os.environ['DEFAULT_TORCH_EXTENSION_PATH'] = f"{local_home}/tmp"
os.environ['TORCH_EXTENSIONS_DIR'] = f"{local_home}/tmp"
os.environ['TORCH_HOME'] = f"{local_home}/tmp"
os.environ['MPLCONFIGDIR'] = f"{local_home}/tmp"
cuda_device = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
torch.hub.set_dir(f"{local_home}/cache")


if __name__ == '__main__':

    seed = 42
    set_global_seed(seed)
    logger = logging.getLogger("pytorch_lightning")

    # 'grayscale' for GS encoding, 'rgb' for SR encoding
    color = 'grayscale'
    train_dir = "grayscale" if color == 'grayscale' else "rainbow"
    train_split = 0.8
    val_split = 1 - train_split

    # dataset train/test splitting
    if os.path.isdir("train_dataset"):
        shutil.rmtree("train_dataset")
        os.mkdir('train_dataset')
        shutil.rmtree("test_set")
        os.mkdir('test_set')
        splitfolders.ratio(train_dir, output="train_dataset", ratio=(train_split, val_split))
        splitfolders.ratio("train_dataset/val", output="test_set", ratio=(.5, .5))
    else:
        os.mkdir('train_dataset')
        os.mkdir('test_set')
        splitfolders.ratio(train_dir, output="train_dataset", ratio=(train_split, val_split))
        splitfolders.ratio("train_dataset/val", output="test_set", ratio=(.5, .5))

    if color == 'grayscale':
        DATA_MEANS, DATA_STD = 0.4968559741973877, 0.19007042050361633
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(root='train_dataset/train', transform=transform)
        loader = DataLoader(dataset, batch_size=len(dataset))
        DATA_MEANS, DATA_STD = mean_std(loader, color)
        del dataset, loader

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(DATA_MEANS, DATA_STD)]) if color != 'grayscale' \
        else transforms.Compose([transforms.Grayscale(1),
                                 transforms.ToTensor(),
                                 transforms.Normalize(DATA_MEANS, DATA_STD)
                                 ])

    train_set = datasets.ImageFolder(root='train_dataset/train', transform=transform)
    val_set = datasets.ImageFolder(root="test_set/train", transform=transform)
    test_set = datasets.ImageFolder(root="test_set/val", transform=transform)

    # input dimensions and hyperparameters
    batch_size = 64
    epochs = 100
    patch_size = 16
    lr = 1e-06
    num_heads = 32
    dropout = 0.6
    weight_decay = 1e-08
    h = 62
    w = 1000
    channels = 1 if color == 'grayscale' else 3

    accelerator: str = "cpu"
    devices: int = 1
    if torch.cuda.is_available():
        accelerator = "gpu"

    train_loader = DataLoader(train_set, batch_size=64 if num_heads == 32 else 128, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64 if num_heads == 32 else 128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64 if num_heads == 32 else 128, shuffle=False)

    log_dir = f"best_models/{color}/PVit_{lr}_{num_heads}_flops"

    csv_logger = CSVLogger(
        save_dir=log_dir,
        name='val_loss_csv',
        version=f'PViT_{color}')

    checkpoint = ModelCheckpoint(log_dir, monitor='val_acc', verbose=False, save_top_k=1, mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(monitor="val_acc", patience=epochs, mode="max")

    trainer = pl.Trainer(
        accelerator='gpu',
        precision='16-mixed',
        check_val_every_n_epoch=1,
        logger=[csv_logger],
        devices=1,
        max_epochs=epochs,
        gradient_clip_val=1.0,
        callbacks=[
            TQDMProgressBar(refresh_rate=3 if batch_size >= 512 else 25),
            checkpoint,
            early_stop

        ],
        enable_progress_bar=True,
    )

    model = EEGVitPretrained(num_heads, channels, lr, weight_decay, dropout)
    input_shape = (1, channels, h, w)
    flops, _, params = calculate_flops(model=model, input_shape=input_shape, print_results=False,
                                       output_as_string=True, print_detailed=False, output_precision=4)
    print("ViT FLOPs:%s   Params:%s \n" % (flops, params))
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    logger.info('Test dataloader')
    test_result = trainer.test(model, dataloaders=test_loader, verbose=True, ckpt_path=checkpoint.best_model_path)
