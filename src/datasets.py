import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import utils as ut
import os


class FrameDataset(Dataset):
    def __init__(self, frames_dir: str, angles_file: str, consecutive_frames: int = 8, 
                 transform : T.Compose = None):
        num_frames = ut.num_of_tensors_in_dir(frames_dir)
        self.frames = ut.load_frames(frames_dir, 0, num_frames)
        self.angles = torch.tensor(ut.load_angles(angles_file, 0, num_frames))
        self.transform = transform
        self.consecutive_frames = consecutive_frames
        
    
    def __getitem__(self, idx):
        frames = torch.stack(self.frames[idx : idx + self.consecutive_frames])
        if self.transform is not None:
            frames = self.transform(frames)
        return (frames, self.angles[idx + self.consecutive_frames])
        
        
    def __len__(self):
        return len(self.frames) - self.consecutive_frames
    

def get_frame_ds(frames_dir: str, consecutive_frames: int = 8, transform: T.Compose = None) -> FrameDataset:
    return FrameDataset(frames_dir, os.path.join(frames_dir, "angles.txt"), consecutive_frames, transform)

def get_dl(frame_ds, batch_size=32, shuffle=False, num_of_workers=3, pin_memory=True) -> DataLoader:
    return DataLoader(frame_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, 
                     num_of_workers=num_of_workers)