import splitfolders
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from utils import get_flops
import shutil
import numpy as np
import random
from keras.callbacks import ModelCheckpoint
from keras import mixed_precision
from models import build_cnn, build_lstm, build_gru
os.environ["TF_KERAS"] = '1'


# 'grayscale' for GS encoding, 'rgb' for SR encoding
color = 'grayscale'
train_dir = "grayscale" if color == 'grayscale' else 'rainbow'
train_split = 0.8
val_split = 1 - train_split

# dataset train/test splitting
if os.path.isdir("train_dataset"):
    shutil.rmtree("train_dataset")
    os.mkdir('train_dataset')
    shutil.rmtree("test_set")
    os.mkdir('test_set')
    splitfolders.ratio(train_dir, output="train_dataset", ratio=(train_split, val_split))
    splitfolders.ratio("train_dataset/val", output="test_set",  ratio=(.5, .5))
else:
    os.mkdir('train_dataset')
    os.mkdir('test_set')
    splitfolders.ratio(train_dir, output="train_dataset",  ratio=(train_split, val_split))
    splitfolders.ratio("train_dataset/val", output="test_set",  ratio=(.5, .5))

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# seed
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# train/val/test generators
train_image_generator = ImageDataGenerator(rescale=1. / 255,)

val_image_generator = ImageDataGenerator(rescale=1. / 255,)

test_image_generator = ImageDataGenerator(rescale=1. / 255,)

# input dimensions
batch_size = 64
epochs = 100
h = 62
w = 1000
channels = 1 if color == 'grayscale' else 3

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory="train_dataset/train",
                                                           shuffle=True,
                                                           target_size=(h, w),
                                                           color_mode=color,
                                                           class_mode='categorical')

val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size,
                                                       directory="test_set/train",
                                                       shuffle=False,
                                                       target_size=(h, w),
                                                       color_mode=color)

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory="test_set/val",
                                                         shuffle=False,
                                                         target_size=(h, w),
                                                         color_mode=color)


models = [build_cnn(h, w, channels, color), build_lstm(h, w, channels, color), build_gru(h, w, channels)]
names = ['CNN_', 'LSTM_', 'GRU_']
val_steps = val_data_gen.samples // batch_size
train_steps = train_data_gen.samples // batch_size
i = 0
for model, model_name in zip(models, names):
    model.summary()
    print(get_flops(model))
    checkpoint = ModelCheckpoint(f"{model_name}best_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max',
                                 save_weights_only=True, save_freq='epoch')

    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=val_steps,
        callbacks=[checkpoint]
    )

    # testing on test set
    model.load_weights(f"{model_name}best_model.hdf5")
    score = model.evaluate(test_data_gen, verbose=1)
    print(f'{model_name} testing score: {score[1]}')
    i = i+1
