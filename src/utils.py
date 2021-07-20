import torch
from torch import nn
import functools
import math
import time
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


#TODO REVISITE THE NEXT 9 FUNCTIONS


#from pytorch implementation of EfficientNet https://github.com/lukemelas/EfficientNet-PyTorch
def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coeff
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


#from pytorch implementation of EfficientNet https://github.com/lukemelas/EfficientNet-PyTorch
def round_filters(filters, global_params):
    multiplier = global_params.width_coeff
    if not multiplier:
        return filters

    new_filters = filters * multiplier
    
    if new_filters < 0.9 * filters:
        new_filters = filters

    return int(new_filters)


def get_width_and_height(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride)

def conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

def conv2d_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int, pl_kernel: int, pl_stride: int):
    return (conv2d(in_channels, out_channels, kernel_size, stride), nn.BatchNorm2d(out_channels), 
            nn.ReLU(), nn.MaxPool2d(pl_kernel, pl_stride))


def conv3d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1)

def conv3d_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int, pl_kernel: int, pl_stride: int):
    return (conv3d(in_channels, out_channels, kernel_size, stride), nn.BatchNorm3d(out_channels), 
            nn.ReLU(), nn.MaxPool3d(pl_kernel, pl_stride))






def create_dir(dir: str) -> None:
    Path(dir).mkdir(exist_ok=True, parents=True)

def create_dirs(dirs: list[str]) -> None:
    for dir in dirs:
        create_dir(dir)

def num_of_tensors_in_dir(dir: str) -> int:
    return len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and 
                os.path.splitext(f)[1] == ".pt"])


def read_angles(angles_file: str) -> list[tuple[float, float]]:
    angles = []
    with open(angles_file, "r") as f:
        for line in f:
            line = line.split()
            angles.append((float(line[0]), float(line[1])))
    return angles


def write_angles_without_nan(output_file: str, angles: list[tuple[float, float]]) -> None:
    with open(output_file, "w") as f:
        for a in angles:
            if math.isnan(a[0]):
                continue
            f.write(f"{a[0]} {a[1]}\n")


def write_angles(output_file: str, angles: list[tuple[float, float]]) -> None:
    with open(output_file, "w") as f:
        for a in angles:
            f.write(f"{a[0]} {a[1]}\n")
            

def load_frames(frames_file: str, start_idx: int, end_idx: int) -> list[torch.Tensor]:
    """
    Load frames in the range start_idx : end_idx (exclusive).
    """
    return [torch.load(os.path.join(frames_file, str(i) + ".pt")) for i in range(start_idx, end_idx)]


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer



def inference_and_save(model, dl, output):
    model.eval()
    with open(output, "w") as f:
        for x, y in dl:
            y_pred = model(x)

            for i in range(y_pred.shape[0]):
                f.write(f"{y_pred[i, 0].item()} {y_pred[i, 1].item()}\n")



def plot_history(history: dict) -> None:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = [i for i in range(len(train_loss))]
    plt.plot(epochs, train_loss, "b-", label="TrainLoss")
    plt.plot(epochs, val_loss, "g-", label="ValidLoss")
    plt.legend(loc="center right", fontsize=12) 
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss)) +1])


def plot_sqrt_history(history: dict) -> None:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = [i for i in range(len(train_loss))]
    plt.plot(epochs, np.sqrt(train_loss), "b-", label="TrainLoss")
    plt.plot(epochs, np.sqrt(val_loss), "g-", label="ValidLoss")
    plt.legend(loc="center right", fontsize=12) 
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("sqrt loss", fontsize=16)
    plt.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss)) +1])


def save_history_img(history: dict, path: str) -> None:
    plot_history(history)
    plt.savefig(path)

def save_sqrt_history_img(history: dict, path: str) -> None:
    plot_sqrt_history(history)
    plt.savefig(path)
