import torch.optim as optim
import lightning as pl
import gc
from typing import List
import logging
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix
import torch
from torch import nn
import transformers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, LSTM, GRU, BatchNormalization,\
    DepthwiseConv2D, AveragePooling2D, SeparableConv2D, Permute, Reshape, Bidirectional
from keras.constraints import max_norm
import tensorflow as tf
logger = logging.getLogger("pytorch_lightning")


class EEGVitPretrained(pl.LightningModule):
    def __init__(self, heads, channels, lr, wd, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=256,
            kernel_size=(1, 31),
            stride=(1, 31),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "google/vit-base-patch16-224-in21k"

        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (62, 32)})
        config.update({'patch_size': (8, 1)})
        config.update({'channels': channels})
        config.update({'num_attention_heads': heads})
        config.update({'num_labels': 3})
        self.save_hyperparameters()

        conv_out = 768 if model_name == "google/vit-base-patch16-224-in21k" else 1024
        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, conv_out, kernel_size=(8, 1),
                                                                           stride=(8, 1), padding=(0, 0), groups=256)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(conv_out, 1000, bias=True),
                                               torch.nn.Dropout(p=dropout),
                                               torch.nn.Linear(1000, 3, bias=True))
        self.ViT = model
        self.lr = lr
        self.weight_decay = wd
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x).logits
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def step(self, batch, mode="train"):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        metric = MulticlassAccuracy(num_classes=3, average='micro').to('cuda')
        acc = metric(preds, labels) 
        batch_dictionary = {
            "loss": loss,
            "acc": acc,
            'preds': preds,
            'targets': labels
        }

        if mode == 'train':
            self.training_step_outputs.append(batch_dictionary)

        elif mode == 'val':
            self.validation_step_outputs.append(batch_dictionary)

        elif mode == 'test':
            self.test_step_outputs.append(batch_dictionary)

        del preds, imgs, labels
        garbage_collection_cuda()
        gc.collect()
        return batch_dictionary

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode="test")

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        self.log_stats(outputs, 'train')
        del outputs
        self.training_step_outputs.clear()
        garbage_collection_cuda()
        gc.collect()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.log_stats(outputs, 'val')
        del outputs
        self.validation_step_outputs.clear()
        garbage_collection_cuda()
        gc.collect()

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.log_stats(outputs, 'test')

        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        conf_matrix = ConfusionMatrix(num_classes=3, task='multiclass').to('cuda')
        cm = conf_matrix(preds, targets).tolist()

        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) - true_pos
        false_neg = np.sum(cm, axis=1) - true_pos

        precision = np.mean(true_pos / (true_pos + false_pos))
        recall = np.mean(true_pos / (true_pos + false_neg))
        f1 = 2 * ((precision * recall) / (precision + recall))

        logger.info(f'Prec: {100 * precision}, Recall: {100 * recall}, F1_score: {100 * f1}')
        logger.info('Confusion matrix:')
        logger.info(cm)
        del outputs
        self.test_step_outputs.clear()
        garbage_collection_cuda()

    def log_stats(self, outputs: List, mode):
        # loss
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        self.log(f"{mode}_loss", epoch_loss, prog_bar=True if mode in ["val", 'train'] else False, sync_dist=True,
                 logger=True)
        # accuracy
        for metric in ["acc"]:
            epoch_metric = sum([x[metric] for x in outputs]) / len(outputs)
            self.log(f"{mode}_{metric}", epoch_metric, prog_bar=True if mode in ["val", "train"] else False,
                     sync_dist=True, logger=True)
            del metric
        del outputs

        
def build_cnn(h, w, channels, color):
    k_size_1 = 3 if color == 'grayscale' else 16
    model = Sequential(name="CNN")
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.add(Conv2D(16, kernel_size=k_size_1, use_bias=True, padding='same',
                     input_shape=(h, w, channels)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, kernel_size=k_size_1, use_bias=False))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.003)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.003)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


def build_lstm(h, w, channels, color):
    k_size_1 = 16
    k_size_2 = 5 if color == 'grayscale' else 16
    units = 50 if color == 'grayscale' else 70

    adam = tf.keras.optimizers.Adam(learning_rate=5e-04)

    model = Sequential(name="LSTM")
    model.add(Conv2D(8, (1, k_size_1), padding='same', input_shape=(h, w, channels), use_bias=False))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D((k_size_2, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(4.)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((1, 4)))

    model.add(SeparableConv2D(96, (1, k_size_2), use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((1, 8)))

    model.add(Permute((2, 1, 3)))
    model.add(Reshape((58, -1)))
    model.add(Bidirectional(LSTM(units, activation='tanh', return_sequences=False)))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_constraint=max_norm(4.)))
    model.add(Dropout(0.5))

    model.add(Dense(256, kernel_constraint=max_norm(4.)))
    model.add(Dropout(0.3))

    model.add(Dense(3, kernel_constraint=max_norm(4.)))
    model.add(Activation('softmax', name='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


def build_gru(h, w, channels):
    k_size_1 = 16
    k_size_2 = 3
    units = 70
    dropout_1 = 0.5
    dropout_2 = 0.3
    adam = tf.keras.optimizers.Adam(learning_rate=5e-04)

    model = Sequential(name="GRU")
    model.add(Conv2D(8, (1, k_size_1), padding='same', input_shape=(h, w, channels), use_bias=False))
    model.add(BatchNormalization())

    model.add(DepthwiseConv2D((k_size_2, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(4.)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((1, 4)))

    model.add(SeparableConv2D(96, (1, k_size_2), use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((1, 8)))

    model.add(Permute((2, 1, 3)))
    model.add(Reshape((60, -1)))
    model.add(Bidirectional(GRU(units, activation='tanh', return_sequences=False)))
    model.add(Dropout(dropout_1))

    model.add(Dense(512, kernel_constraint=max_norm(4.)))
    model.add(Dropout(dropout_1))

    model.add(Dense(256, kernel_constraint=max_norm(4.)))
    model.add(Dropout(dropout_2))

    model.add(Dense(3, kernel_constraint=max_norm(4.)))
    model.add(Activation('softmax', name='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model
