import torch
from torchvision import transforms as T
import cv2 as cv
import os
import numpy as np
import utils as ut


def from_video_to_tensors(video_path: str, transform: T.Compose) -> torch.Tensor:
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
    return torch.stack(frames)



def setup_videos(video_paths: list[str], output_paths: list[str], angles_paths: list[str], selected_frames_paths: list[str],
                 transform: T.Compose) -> None:
    """
    Convert every video in video_paths and pair with the corresponding angles. Save the tensors in output_paths.
    """
    for video_path, output_path, angles_path, selected_frames_path in zip(video_paths, output_paths, angles_paths, selected_frames_paths):
        print(f"Start conversion of {video_path}.")
        
        frames = from_video_to_tensors(video_path, transform)
        angles = np.loadtxt(angles_path, dtype=np.float64)
        selected_frames = np.reshape(np.loadtxt(selected_frames_path, dtype=np.int32), (-1, 2))
        for i in range(selected_frames.shape[0]):
            k, j = selected_frames[i]
            angles_without_nan = np.invert(np.isnan(angles[k:j, 0])) 
            torch.save(frames[k : j][angles_without_nan], os.path.join(output_path, str(i) + ".pt"))
            np.savetxt(os.path.join(output_path, str(i) + ".txt"), angles[k:j][angles_without_nan])



def setup_videos_single_tensor(video_paths: list[str], output_paths: list[str], angles_paths: list[str], selected_frames_paths: list[str],
                 transform: T.Compose) -> None:
    """
    Convert every video in video_paths and pair with the corresponding angles. Save the tensors in output_paths.
    """
    for video_path, output_path, angles_path, selected_frames_path in zip(video_paths, output_paths, angles_paths, selected_frames_paths):
        print(f"Start conversion of {video_path}.")
        
        frames = from_video_to_tensors(video_path, transform)
        angles = np.loadtxt(angles_path, dtype=np.float64)
        selected_frames = np.reshape(np.loadtxt(selected_frames_path, dtype=np.int32), (-1, 2))
        for i in range(selected_frames.shape[0]):
            k, j = selected_frames[i]
            angles_without_nan = np.invert(np.isnan(angles[k:j, 0])) 
            tensors = frames[k : j][angles_without_nan]
            p = os.path.join(output_path, str(i))
            ut.create_dir(p)
            for u in range(tensors.size(0)):
                torch.save(tensors[u].clone(), os.path.join(p, str(u) + ".pt"))

            np.savetxt(os.path.join(p, "angles.txt"), angles[k:j][angles_without_nan])
    

def convert_test_video(video_path, output, angles_path, transform):
    print(f"Convert test video {video_path}")

    frames = from_video_to_tensors(video_path, transform)
    angles = np.loadtxt(angles_path, dtype=np.float64)
    
    torch.save(frames, os.path.join(output, "test_video.pt"))
    np.savetxt(os.path.join(output, "angles.txt"), angles)