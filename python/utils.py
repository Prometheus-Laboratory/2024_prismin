import tensorflow as tf
from tensorflow.python.profiler import model_analyzer, option_builder
import numpy as np
import lightning as pl
import torch
import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


def mean_std(loader, color):
    images, labels = next(iter(loader))
    if color == 'grayscale':
        mean, std = torch.mean(images), torch.std(images)
    else:
        mean, std = torch.mean(images, dim=[0, 2, 3]), torch.std(images, dim=[0, 2, 3])
    return mean, std


def get_flops(model):
    input_signature = [tf.TensorSpec(shape=(1, *params.shape[1:]), dtype=params.dtype, name=params.name)
                       for params in model.inputs]
    forward_graph = tf.function(model, input_signature).get_concrete_function().graph
    options = option_builder.ProfileOptionBuilder.float_operation()
    graph_info = model_analyzer.profile(forward_graph, options=options)
    gflops = (graph_info.total_float_ops // 2)*1e-09
    return gflops
