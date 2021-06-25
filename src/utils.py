import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import functools
import cv2 as cv
import math
from multiprocessing import Pool
import os
import shutil
from pathlib import Path
import time
import collections


BlockArgs = collections.namedtuple("BlockArgs", [
            "num_repeat", "in_channels", "out_channels", "kernel_size", "stride",
            "activation_fun", 
])


GlobalParams = collections.namedtuple("GlobalParams", [
    "width_coeff", "depth_coeff", "image_size",
    "conv_layers", #first layers of the conv 
    "final_pooling_layer",
    "num_recurrent_layers", "hidden_dim", "rec_dropout", #recurrent networks for temporal dependencies 
    "dropout",
    "fc_layers", #fully connected layers (tuple of ints)
])


GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

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

def conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

def conv_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int, pl_kernel: int, pl_stride: int):
    return (conv(in_channels, out_channels, kernel_size, stride), nn.BatchNorm2d(out_channels), 
            nn.ReLU(), nn.MaxPool2d(pl_kernel, pl_stride))


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, output, target):
        return torch.sqrt(F.mse_loss(output, target) + 1e-6)



#UTILITY 

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


def load_frames(path: str, start_idx: int, end_idx: int) -> list[torch.Tensor]:
    return [torch.load(os.path.join(path, str(i) + ".pt")) for i in range(start_idx, end_idx)]


def load_angles(path: str, start_idx: int, end_idx: int) -> list[tuple[float, float]]:
    return read_angles(path)[start_idx:end_idx]


def from_video_to_tensors(video_path: str, transform: T.Compose) -> list[torch.Tensor]:
    """
    Convert every frame of the video into a tensor.
    
    video_path: the path of the video.
    transform: transformation to apply to every frame.
    return: list of frames transformed.
    """
    frames = []
    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()
    return frames


def pair_frames_angles(output_path: str, angles_file: str, frames: list[torch.Tensor]) -> None:
    """
    Pair every frame with the corresponding angles. If the angles are nan, skip the frame.
    """
    angles = read_angles(angles_file)
    i = 0
    
    for a in angles:
        if math.isnan(a[0]):
            continue
        torch.save(frames[i], os.path.join(output_path, str(i) + ".pt"))
        i += 1
    
    write_angles_without_nan(os.path.join(output_path, "angles.txt"), angles)

    
def _read_and_pair(video_path: str, output_path: str, angles_file: str, transform: T.Compose) -> None:
    frames = from_video_to_tensors(video_path, transform)
    pair_frames_angles(output_path, angles_file, frames)
    

@timer
def setup_videos(video_paths: list[str], output_paths: list[str], angles_paths: list[str], 
                 transform: T.Compose, num_of_cpu : int = 2) -> None:
    """
    Convert every video in video_paths and pair with the corresponding angles. Save the tensors in output_paths.
    """
    f = functools.partial(_read_and_pair, transform=transform)
    with Pool(processes=num_of_cpu) as p:
        p.starmap(f, list(zip(video_paths, output_paths, angles_paths)))
        

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


def split_video(video_dir: str, train_dir: str, valid_dir: str, test_dir: str,
                train_split: float, test_split: float) -> None:
    i = 0
    w = 0
    k = 0
    
    num_of_frames = num_of_tensors_in_dir(video_dir)
    train_len = math.floor(train_split * num_of_frames)
    test_len = math.floor(test_split * num_of_frames)
    valid_len = num_of_frames - train_len - test_len
    
    train_start_idx = 0
    train_end_idx = train_len
    valid_start_idx = train_len
    valid_end_idx = train_len + valid_len
    test_start_idx = train_len + valid_len
    test_end_idx = num_of_frames
    
    train_angles = []
    valid_angles = []
    test_angles = []
    
    angles_file = os.path.join(video_dir, "angles.txt")
    angles = read_angles(angles_file)
    
    train_angles += angles[train_start_idx : train_end_idx]
    valid_angles += angles[valid_start_idx : valid_end_idx]
    test_angles += angles[test_start_idx : test_end_idx]
    
    train_dir = os.path.join(train_dir, os.path.split(video_dir)[-1])
    create_dir(train_dir)
    for j in range(train_start_idx, train_end_idx):
        shutil.move(os.path.join(video_dir, str(j) + ".pt"), os.path.join(train_dir, str(i) + ".pt"))
        i += 1
    
    valid_dir = os.path.join(valid_dir, os.path.split(video_dir)[-1])
    create_dir(valid_dir)
    for j in range(valid_start_idx, valid_end_idx):
        shutil.move(os.path.join(video_dir, str(j) + ".pt"), os.path.join(train_dir, str(k) + ".pt"))
        k += 1
        
    test_dir = os.path.join(test_dir, os.path.split(video_dir)[-1])
    create_dir(test_dir)
    for j in range(test_start_idx, test_end_idx):
        shutil.move(os.path.join(video_dir, str(j) + ".pt"), os.path.join(train_dir, str(w) + ".pt"))
        w += 1
    
    write_angles(os.path.join(train_dir, "angles.txt"), train_angles)
    write_angles(os.path.join(valid_dir, "angles.txt"), valid_angles)
    write_angles(os.path.join(test_dir, "angles.txt"), test_angles)
    

@timer
def split_train_valid_test(video_dirs: list[str], output_dir: str, train_split: float, test_split: float, 
                            num_of_cpu: int =2) -> None:
    """
    """
    train_dir = os.path.join(output_dir, "basic_train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")
    
    create_dir(train_dir)
    create_dir(valid_dir)
    create_dir(test_dir)
    
    f = functools.partial(split_video, train_dir=train_dir, valid_dir=valid_dir, test_dir=test_dir, 
                          train_split=train_split, test_split=test_split)
    with Pool(processes=num_of_cpu) as p:
        p.map(f, video_dirs)

        
def augment_video(input_dir: str, output_dir: str, transform: T.Compose) -> None:
    for i in range(num_of_tensors_in_dir(input_dir)):
        file = str(i) + ".pt"
        frame = torch.load(os.path.join(input_dir, file))
        frame = transform(frame)
        torch.save(frame, os.path.join(output_dir, file))
    write_angles(os.path.join(output_dir, "angles.txt"), read_angles(os.path.join(input_dir, "angles.txt")))
        

@timer
def augment_videos(input_dirs: list[str], output_dirs : list[str], transform: T.Compose, num_of_cpu: int = 2) -> None:
    f = functools.partial(augment_video, transform=transform)
    with Pool(processes=num_of_cpu) as p:
        p.starmap(f, list(zip(input_dirs, output_dirs)))

        
def create_dir(dir: str) -> None:
    Path(dir).mkdir(exist_ok=True, parents=True)

def create_dirs(dirs: list[str]) -> None:
    for dir in dirs:
        create_dir(dir)


#UTILITY


#TRANSFORMATIONS

class BGRtoRGB:
    def __call__(self, frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
class CropHeight:
    def __init__(self, y1, y2):
        self.y1 = y1
        self.y2 = y2
        
    def __call__(self, frame):
        return frame[:, self.y1 : self.y2, :]

#TRANSFORMATIONS