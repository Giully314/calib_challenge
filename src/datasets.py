import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import utils as ut
import os
import numpy as np


class ConsecutiveFramesDataset(Dataset):
    """
    Returns a sequence of frames.
    video_path: path of the tensor (converted from a video)
    angles_path: path of the angles associated to the video.
    consecutive_frames: how many consecutive frames to consider.
    skips: how many frames skips between 2 consecutive frames.
    transform: transformation applied before returning the frames.
    """

    def __init__(self, video_path: str, angles_path: str, consecutive_frames: int = 3, skips: int = 2, 
                 transform : T.Compose = None):
        self.frames = torch.load(video_path)
        # self.angles = torch.from_numpy(np.rad2deg(np.loadtxt(angles_path, dtype=np.float64), dtype=np.float64).astype(np.float32))
        self.angles = torch.from_numpy(np.loadtxt(angles_path, np.float32))
        self.transform = transform
        self.consecutive_frames = consecutive_frames
        self.skips = skips
        
    
    def __getitem__(self, idx):
        frames = self.frames[idx : idx + (self.consecutive_frames * self.skips) : self.skips]
        angles = self.angles[idx : idx + (self.consecutive_frames * self.skips) : self.skips].reshape(-1)
        # angles = self.angles[idx + (self.consecutive_frames * self.skips)]
        #TODO add the transformation to angles too, if the frame is flipped.
        if self.transform is not None:
            frames = self.transform(frames)
        return (frames, angles)
        
        
    def __len__(self):
        return self.frames.shape[0] - (self.consecutive_frames * self.skips) + 1


def get_consecutive_frames_ds(video_path: str, angles_path: str, consecutive_frames: int = 3, skips:int = 2, transform: T.Compose = None) -> ConsecutiveFramesDataset:
    return ConsecutiveFramesDataset(video_path, angles_path, consecutive_frames, skips, transform)    



class FrameDataset(Dataset):
    def __init__(self, video_path, angles_file, transform = None):
        self.frames = torch.load(video_path)
        self.angles = torch.from_numpy(np.loadtxt(angles_file, np.float32))
        self.transform = transform

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform is not None:
            frame = self.transform(frame)
        return (frame, self.angles[idx])

    def __len__(self):
        return self.frames.shape[0]


def get_frame_ds(video_path: str, angles_path: str, transform: T.Compose = None) -> FrameDataset:
    return FrameDataset(video_path, angles_path, transform) 





