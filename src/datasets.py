import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from itertools import accumulate
import random
import os
import utils as ut

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
                 transforms : list[T.Compose] = None):
        self.frames = torch.load(video_path)
        self.angles = torch.from_numpy(np.loadtxt(angles_path, np.float32))
        self.transforms = transforms
        self.consecutive_frames = consecutive_frames
        self.skips = skips
        
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.frames[idx : idx + (self.consecutive_frames * self.skips) : self.skips]
        angles = self.angles[idx : idx + (self.consecutive_frames * self.skips) : self.skips].reshape(-1)
        
        #TODO add the transformation to angles too, if the frame is flipped.
        if self.transforms is not None:
            n = len(self.transforms)
            i = random.randint(0, n - 1)
            frames = self.transforms[i](frames)
        
        return (frames, angles)
        
        
    def __len__(self) -> int:
        return self.frames.shape[0] - (self.consecutive_frames * self.skips) + 1


def get_consecutive_frames_ds(video_path: str, angles_path: str, consecutive_frames: int = 3, skips:int = 2, transforms: list[T.Compose] = None) -> ConsecutiveFramesDataset:
    return ConsecutiveFramesDataset(video_path, angles_path, consecutive_frames, skips, transforms)    



class DiskConsecutiveFramesDataset(Dataset):
    """
    Returns a sequence of frames (from disk).
    video_path: path of the tensor (converted from a video)
    angles_path: path of the angles associated to the video.
    consecutive_frames: how many consecutive frames to consider.
    skips: how many frames skips between 2 consecutive frames.
    transform: transformation applied before returning the frames.
    """

    def __init__(self, video_path: str, consecutive_frames: int = 3, skips: int = 2, 
                 transforms : list[T.Compose] = None):
        self.video_path = video_path
        self.angles = torch.from_numpy(np.loadtxt(os.path.join(video_path, "angles.txt"), np.float32))
        self.transforms = transforms
        self.consecutive_frames = consecutive_frames
        self.skips = skips
        self.len = ut.num_of_tensors_in_dir(self.video_path)
        
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = []
        for i in range(0, self.consecutive_frames * self.skips, self.skips):
            frames.append(torch.load(os.path.join(self.video_path, str(idx + i) + ".pt")))

        frames = torch.stack(frames)
        angles = self.angles[idx : idx + (self.consecutive_frames * self.skips) : self.skips].reshape(-1)
      
        #TODO add the transformation to angles too, if the frame is flipped.
        if self.transforms is not None:
            n = len(self.transforms)
            i = random.randint(0, n - 1)
            frames = self.transforms[i](frames)
        
        return (frames, angles)
        
    def __len__(self) -> int:
        return self.len - (self.consecutive_frames * self.skips) + 1

def get_disk_consecutive_frames_ds(video_path: str, consecutive_frames: int = 3, skips:int = 2, transforms: list[T.Compose] = None) -> ConsecutiveFramesDataset:
    return DiskConsecutiveFramesDataset(video_path, consecutive_frames, skips, transforms)   


class DiskVideoDataset(Dataset):
    """
    This dataset provides a way to take portions of consecutive frames of different videos while training. 
    This is usefull when using the dataloader with the option of shuffle. 
    """
    def __init__(self, videos_files: list[str], consecutive_frames: int = 5, skips: int = 1, 
                    transforms: list[T.Compose] = None):
        
        self.datasets = [get_disk_consecutive_frames_ds(video_path, consecutive_frames, skips, transforms) 
                        for video_path in videos_files]

        #TODO: write better this explanation.
        #partial sum of length of the videos. This will serve to "translate" the index passed from the dataloader to the corresponding
        #video. initial=0 so i can safely subtract the previous value to get the real index. 
        self.idxs = list(accumulate([len(ds) for ds in self.datasets], initial=0)) 
        
        self.len = sum([len(ds) for ds in self.datasets])
        

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        #I could use a binary search but it's pointless because the size of this list is maximum 15.
        for i in range(len(self.idxs)):
            if idx < self.idxs[i]:
                break

        video = i - 1 #subtract 1 from the index beacuse the list has an extra element (the first) that is 0.
        idx = idx - self.idxs[i - 1]
        
        return self.datasets[video][idx]

    def __len__(self) -> int:
        return self.len 



class VideoDataset(Dataset):
    """
    This dataset provides a way to take portions of consecutive frames of different videos while training. 
    This is usefull when using the dataloader with the option of shuffle. 
    """
    def __init__(self, videos_files: list[str], angles_files: list[str], consecutive_frames: int = 5, skips: int = 1, 
                    transforms: list[T.Compose] = None):
        
        self.datasets = [get_consecutive_frames_ds(video_path, angles_path, consecutive_frames, skips, transforms) 
                        for video_path, angles_path in zip(videos_files, angles_files)]

        #TODO: write better this explanation.
        #partial sum of length of the videos. This will serve to "translate" the index passed from the dataloader to the corresponding
        #video. initial=0 so i can safely subtract the previous value to get the real index. 
        self.idxs = list(accumulate([len(ds) for ds in self.datasets], initial=0)) 
        
        self.len = sum([len(ds) for ds in self.datasets])
        

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        #I could use a binary search but it's pointless because the size of this list is maximum 10.
        for i in range(len(self.idxs)):
            if idx < self.idxs[i]:
                break

        video = i - 1 #subtract 1 from the index beacuse the list has an extra element (the first) that is 0.
        idx = idx - self.idxs[i - 1]
        
        return self.datasets[video][idx]

    def __len__(self) -> int:
        return self.len 





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





