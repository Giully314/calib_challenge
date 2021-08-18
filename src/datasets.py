import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import utils as ut
import os
import numpy as np


class ConsecutiveFramesDataset(Dataset):
    def __init__(self, frames_dir: str, angles_file: str, consecutive_frames: int = 3, step: int = 2, 
                 transform : T.Compose = None):
        num_frames = ut.num_of_tensors_in_dir(frames_dir)
        self.frames = torch.stack(ut.load_frames(frames_dir, 0, num_frames))
        self.angles = torch.tensor(np.loadtxt(angles_file, dtype=np.float32))
        self.transform = transform
        self.consecutive_frames = consecutive_frames
        self.step = step
        
    
    def __getitem__(self, idx):
        frames = self.frames[idx : idx + (self.consecutive_frames * self.step) : self.step]
        angles = torch.stack(self.angles[idx : idx + (self.consecutive_frames * self.step) : self.step])
        if self.transform is not None:
            frames = self.transform(frames)
        return (frames, angles)
        
        
    def __len__(self):
        return len(self.frames) - (self.consecutive_frames * self.step) + 1


def get_consecutive_frames_ds(frames_dir: str, consecutive_frames: int = 3, step:int = 2, transform: T.Compose = None) -> ConsecutiveFramesDataset:
    return ConsecutiveFramesDataset(frames_dir, os.path.join(frames_dir, "angles.txt"), consecutive_frames, step, transform)    



class FrameDataset(Dataset):
    def __init__(self, frames_dir, angles_file, transform = None):
        self.frames = torch.stack(ut.load_frames(frames_dir, 0, ut.num_of_tensors_in_dir(frames_dir)))
        self.angles = torch.tensor(np.loadtxt(angles_file, np.float32))
        # self.frames_dir = frames_dir
        self.transform = transform

    def __getitem__(self, idx):
        # torch.load(os.path.join(self.frames_dir, str(idx) + ".pt"))
        # return (torch.load(os.path.join(self.frames_dir, str(idx) + ".pt")), self.angles[idx]) 
        frame = self.frames[idx]
        if self.transform is not None:
            frame = self.transform(frame)
        return (frame, self.angles[idx])

    def __len__(self):
        return len(self.angles)


def get_frame_ds(frames_dir: str, transform: T.Compose = None) -> FrameDataset:
    return FrameDataset(frames_dir, os.path.join(frames_dir, "angles.txt"), transform) 



def get_dl(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, 
                     num_workers=num_workers)



