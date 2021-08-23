from models import CalibModel 
from torch.utils.data import DataLoader
import torch
from datasets import get_consecutive_frames_ds
import training as trn
from custom_transform import CropVideo
import os 
import numpy as np
from torch import nn
import random 
import hydra
from omegaconf import DictConfig, OmegaConf

#TODO add verbose mode with timing and ecc...

@hydra.main(config_path="config", config_name="nvidia_training.yaml")
def main(cfg: DictConfig):
    args = cfg["train"]

    if args.deterministic:
        seed = 1729
        def reset_rand_seed(): 
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        reset_rand_seed()
        #torch.use_deterministic_algorithms(True)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_size = tuple(args.frame_size)
    height, width = img_size

    verbose = args.verbose

    batch_size = args.batch_size
    train_workers = args.train_workers
    valid_workers = args.valid_workers
    test_workers = args.test_workers
    shuffle = args.shuffle
    persistent_workers = args.persistent_workers

    data_dir = args.data
    videos = args.train_videos
    videos_parts = args.videos_parts
    train_videos = [os.path.join(data_dir, os.path.join(str(video), str(part) + ".pt")) 
                    for video, video_parts in zip(videos, videos_parts) for part in video_parts]
    train_angles = [os.path.join(data_dir, os.path.join(str(video), str(part) + ".txt")) 
                            for video, video_parts in zip(videos, videos_parts) for part in video_parts]
    
    consecutive_frames = args.consecutive_frames
    skips = args.skips
    
    trf_crop = CropVideo(int(width * 0.2), int(width - width * 0.2), int(height *0.45), int(height - height * 0.2))
    
    img_size = (trf_crop.y2 - trf_crop.y1, trf_crop.x2 - trf_crop.x1)

    train_dss = [get_consecutive_frames_ds(video_path, angles_path, consecutive_frames, skips, trf_crop) 
                for video_path, angles_path in zip(train_videos, train_angles)]
    
    # ds = train_dss[0]
    # ds.frames = ds.frames[0:256]
    # ds.angles = ds.angles[0:256]

    train_dls = [
        DataLoader(train_ds, batch_size,
        num_workers=train_workers, shuffle=shuffle, pin_memory=True, persistent_workers=persistent_workers) 
        for train_ds in train_dss]

    


    # if args.dataset == "FrameDataset":
    #     train_datasets = [get_frame_ds(frame_dir) for frame_dir in train_dirs]
    #     valid_datasets = [get_frame_ds(frame_dir) for frame_dir in valid_dirs]
    # elif args.dataset == "RangeFrameDataset":
    #     train_datasets = [get_range_frame_ds(frame_dir) for frame_dir in train_dirs]
    #     valid_datasets = [get_range_frame_ds(frame_dir) for frame_dir in valid_dirs]
    # elif args.dataset == "ConsecutiveFrameDataset":
    #     train_datasets = [get_consecutive_frames_ds(frame_dir) for frame_dir in train_dirs]
    #     valid_datasets = [get_frame_ds(frame_dir) for frame_dir in valid_dirs]

    valid_dl = None
    if args.valid_model:
        valid_dir = os.path.join(data_dir, args.valid_data)
        # valid_dirs = [os.path.join(valid_dir, str(video)) for video in videos]
        # valid_angles_files = [os.path.join(vd, "angles.txt") for vd in valid_dirs]
        # valid_datasets = [get_frame_ds(frame_dir) for frame_dir in valid_dirs]
        # valid_dl = [DataLoader(valid_ds, batch_size, 
        #     num_workers=valid_workers, shuffle=False, pin_memory=True, persistent_workers=persistent_workers) 
        #     for valid_ds in valid_datasets]
    

    model = CalibModel(img_size, consecutive_frames)
    model.to(dev)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN SHAPE OUT: {model.cnn_shape_out}")
    print(f"Number of parameters {pytorch_total_params}")
    #TODO add support for passing other parameters to the optimizer
    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    loss = nn.MSELoss()

    history = trn.History(args.history_out, videos, args.valid_video, args.test_video, model, opt, loss, None, batch_size)
    epochs = args.epochs
    
    trn.fit(epochs, history, train_dls, valid_dl, dev, verbose=verbose)

    if args.save_history:
        history.save_training_info()
    
    
    if args.test_model:
        #load test data here
        test_dl = None
        history.test_model(test_dl, dev)   
    
    
    if args.save_model:
        history.save_model()

if __name__ == "__main__":
    main()