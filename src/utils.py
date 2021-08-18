import dataclasses
import torch
from torch import nn
import functools
import math
import time
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
import shutil


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
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

def conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False)

def conv2d_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int, pl_kernel: int, pl_stride: int):
    return (conv2d(in_channels, out_channels, kernel_size, stride), nn.BatchNorm2d(out_channels), 
            nn.ReLU(), nn.MaxPool2d(pl_kernel, pl_stride))





def create_dir(dir: str) -> None:
    Path(dir).mkdir(exist_ok=True, parents=True)

def create_dirs(dirs: list[str]) -> None:
    for dir in dirs:
        create_dir(dir)

def delete_dir(dir: str) -> None: 
    shutil.rmtree(dir)

def delete_dirs(dirs: list[str]) -> None:
    for dir in dirs:
        shutil.rmtree(dir)

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

@dataclass
class Timer:
    s: float = 0
    e: float = 0

    def start(self):
        self.s = time.time()

    def end(self):
        self.e = time.time()

    def __str__(self):
        total_time = self.e - self.s 
        return f"{(total_time//60):.0f}m {(total_time % 60):.0f}s"
    


def inference_and_save(model, dl, output, dev = torch.device("cpu")):
    model.eval()
    with open(output, "w") as f:
        for x, y in dl:
            y_pred = model(x.to(dev))

            for i in range(y_pred.shape[0]):
                f.write(f"{y_pred[i, 0].item()} {y_pred[i, 1].item()}\n")



def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))


def eval_angles(gt_file, test_file):
    gt = np.loadtxt(gt_file)
    zero_mse = get_mse(gt, np.zeros_like(gt))

    test = np.loadtxt(test_file)
    mse = get_mse(gt, test)

    return mse, zero_mse



#visualize cnn filters weights (code from: https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch) 
def visualize_cnn_filters(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torch.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))



#class for register activation map
@dataclass 
class ActivationMapHook:
    activation: dict = field(default_factory=dict)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook


    def __getitem__(self, idx):
        return self.activation[idx]
    
    def __setitem__(self, idx, new_value):
        self.activation[idx] = new_value