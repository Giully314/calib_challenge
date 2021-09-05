from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import os
import numpy as np
import utils as ut
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from custom_transform import Crop


def read_tensors_angles(path):
    tensors = []
    for i in range(ut.num_of_tensors_in_dir(path)):
        tensors.append(torch.load(os.path.join(path, str(i) + ".pt")))

    #i read the angles because if i flip the frame i need to transform the angles.
    angles = np.loadtxt(os.path.join(path, "angles.txt"), dtype=np.float64)

    return tensors, angles


#NOTE: this code is not optimized.

@hydra.main(config_path="config")
def augment(cfg: DictConfig):
    args = cfg["setup_conversion"]["augment"]
    
    videos = [os.path.join(str(video), str(part)) for video, parts in zip(args.videos, args.videos_parts) 
                                            for part in parts]
    videos_path = [os.path.join(args.data_dir, video) for video in videos]
    

    trf = cfg["setup_conversion"]["transformations"]
    if trf.transforms is not None:
        if trf.color_jitter:
            trf_jitter = T.ColorJitter(tuple(trf.brightness), tuple(trf.contrast), trf.saturation, trf.hue)
        if trf.rotation:
            trf_rotation = T.RandomRotation(tuple(trf.degrees), interpolation=InterpolationMode.BILINEAR)
        if trf.translate:
            trf_translate = T.RandomAffine(0, translate=tuple(trf.translations), interpolation=InterpolationMode.BILINEAR)
        if trf.crop:
            trf_crop = Crop(*list(trf.crop_args))
        
        t = []
        for transform in trf.transforms:
            if transform == "rotation":
                t.append(trf_rotation)
            if transform == "color_jitter":
                t.append(trf_jitter)
            if transform == "translate":
                t.append(trf_translate)
            if transform == "crop":
                t.append(trf_crop)
        
        t = T.Compose(t)

    videos_aug_path = [os.path.join(args.output_dir, str(video)) for video in videos] 
    ut.create_dirs(videos_aug_path)

    for video in videos_path:
        print(video)
    
    for video in videos_aug_path:
        print(video)
        

    for video_path, out_path in zip(videos_path, videos_aug_path):
        print(f"Augmenting {video_path}")
        tensors, angles = read_tensors_angles(video_path)

        for i, tensor in enumerate(tensors):
            torch.save(t(tensor), os.path.join(out_path, str(i) + ".pt"))
        
        np.savetxt(os.path.join(out_path, "angles.txt"), angles)





if __name__ == "__main__":
    augment()