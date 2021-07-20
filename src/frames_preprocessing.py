import torch
from torchvision import transforms as T
import functools
import cv2 as cv
import math
from multiprocessing import Pool
import os
import shutil
from utils import (create_dir, num_of_tensors_in_dir, read_angles, write_angles, write_angles_without_nan)




def from_video_to_tensors(video_path: str, transform: T.Compose) -> list[torch.Tensor]:
    """
    Convert every frame of the video into a tensor and apply a transformation.
    
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
        
        frames.append(transform(frame))
    
    cap.release()
    return frames


def pair_frames_angles_without_nan(output_path: str, angles_file: str, frames: list[torch.Tensor]) -> None:
    """
    Pair every frame with the corresponding angles. NaN angles are eliminated with the corresponding frames.
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
    pair_frames_angles_without_nan(output_path, angles_file, frames)


def setup_videos(video_paths: list[str], output_paths: list[str], angles_paths: list[str], 
                 transform: T.Compose, num_of_cpu : int = 2) -> None:
    """
    Convert every video in video_paths and pair with the corresponding angles. Save the tensors in output_paths.
    """
    f = functools.partial(_read_and_pair, transform=transform)
    with Pool(processes=num_of_cpu) as p:
        p.starmap(f, list(zip(video_paths, output_paths, angles_paths)))

    # for video_path, output_path, angles_path in zip(video_paths, output_paths, angles_paths):
    #     _read_and_pair(video_path, output_path, angles_path, transform)
        


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
    
    #sometimes there isn't a validation set
    if k > 0:
        write_angles(os.path.join(valid_dir, "angles.txt"), valid_angles)
    
    #sometimes there isn't a test set
    if w > 0:
        write_angles(os.path.join(test_dir, "angles.txt"), test_angles)


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
        torch.save(transform(frame), os.path.join(output_dir, file))
    write_angles(os.path.join(output_dir, "angles.txt"), read_angles(os.path.join(input_dir, "angles.txt")))
        


def augment_videos(input_dirs: list[str], output_dirs : list[str], transform: T.Compose, num_of_cpu: int = 2) -> None:
    f = functools.partial(augment_video, transform=transform)
    with Pool(processes=num_of_cpu) as p:
        p.starmap(f, list(zip(input_dirs, output_dirs)))


def copy_last_n_frames(src, dest, n):
    src_num = num_of_tensors_in_dir(src)
    dest_num = num_of_tensors_in_dir(dest)

    
    for i in range(dest_num - 1, -1, -1):
        shutil.move(os.path.join(dest, str(i) + ".pt"), 
                    os.path.join(dest, str(i + n) + ".pt"))
        
    src_angles = read_angles(os.path.join(src, "angles.txt"))[-5:]
    dest_angles = read_angles(os.path.join(dest, "angles.txt"))

    write_angles(os.path.join(dest, "aug_angles.txt"), src_angles + dest_angles)


    j = 0
    for i in range(src_num - n, src_num):
        shutil.copy(os.path.join(src, str(i) + ".pt"),
                    os.path.join(dest, str(j) + ".pt"))
        j += 1


def merge_frames(input_dirs, output_dir):
    i = 0
    angles = []
    for input_dir in input_dirs:
        for j in range(num_of_tensors_in_dir(input_dir)):
            shutil.move(os.path.join(input_dir, str(j) + ".pt"), os.path.join(output_dir, str(i) + ".pt"))
            i += 1

        angles += read_angles(os.path.join(input_dir, "angles.txt"))
    
    write_angles(angles, os.path.join(output_dir, "angles.txt"))