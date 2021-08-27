from models import CalibModel, CalibParams
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
from utils import Timer

#TODO add verbose mode with timing and ecc...
#TODO ADD LOGGING

@hydra.main(config_path="config")
def main(cfg: DictConfig):
    args = cfg["training"]["train"]

    verbose = args.verbose
    
    data_dir = args.data
    videos = args.train_videos
    videos_parts = args.videos_parts

    valid_model = args.valid_model
    valid_video = args.valid_video
    valid_part = args.valid_part

    test_model = args.test_model
    test_video = args.test_video
    test_part = args.test_part
    
    deterministic = args.deterministic

    img_size = tuple(args.frame_size)
    
    consecutive_frames = args.consecutive_frames
    skips = args.skips

    batch_size = args.batch_size
    shuffle = args.shuffle
    persistent_workers = args.persistent_workers
    train_workers = args.train_workers
    valid_workers = args.valid_workers
    test_workers = args.test_workers
    

    timer = Timer()

    opt = args.opt
    lr = args.lr

    epochs = args.epochs

    training_info_dir = args.training_info_dir
    save_history = args.save_history
    save_model = args.save_model
    activation_maps = args.activation_maps

    if deterministic:
        seed = 1729
        def reset_rand_seed(): 
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        reset_rand_seed()
        #torch.use_deterministic_algorithms(True)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
    print(f"DEVICE {dev}")

    height, width = img_size


    train_videos = [os.path.join(data_dir, os.path.join(str(video), str(part) + ".pt")) 
                    for video, video_parts in zip(videos, videos_parts) for part in video_parts]
    train_angles = [os.path.join(data_dir, os.path.join(str(video), str(part) + ".txt")) 
                            for video, video_parts in zip(videos, videos_parts) for part in video_parts]
    

    #TODO implement a config file for transformations. 
    trf_crop = CropVideo(int(width * 0.3), int(width - width * 0.3), int(height *0.45), int(height - height * 0.25))
    
    img_size = (trf_crop.y2 - trf_crop.y1, trf_crop.x2 - trf_crop.x1)

    print("Loading datasets...", end=" ")
    timer.start()
    train_dss = [get_consecutive_frames_ds(video_path, angles_path, consecutive_frames, skips, trf_crop) 
                for video_path, angles_path in zip(train_videos, train_angles)]
    timer.end()
    print(f"{timer}")




    if args.range is not None:
        start = args.range[0]
        end = args.range[1]
        for ds in train_dss:
            ds.frames = ds.frames[start:end]
            ds.angles = ds.angles[start:end]

    train_dls = [
        DataLoader(train_ds, batch_size,
        num_workers=train_workers, shuffle=shuffle, pin_memory=True, persistent_workers=persistent_workers) 
        for train_ds in train_dss]


    valid_dl = None
    if valid_model:
        pass
    
    calib_params = CalibParams(img_size, consecutive_frames, args.lstm_hidden_size, args.lstm_num_layers, args.linear_layers)
    model = CalibModel(calib_params)
    model.to(dev)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #TODO add these informations to the History object and save on the file.
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

    history = trn.History(args.training_info_dir, videos, args.valid_video, model, opt, loss, None, batch_size)
    
    trn.fit(epochs, history, train_dls, valid_dl, dev, verbose=verbose)

    visualization = trn.ModelVisualization(training_info_dir, model, dev, verbose)
    if activation_maps is not None:
        for layer_name in args.activation_maps:
            visualization.register_activation_map(layer_name)

        #NOTE: TECHNICALLY THE VISUALIZATION OF THE ACTIVATION MAPS SHOULD BE DONE ON VALID/TEST SET. FOR NOW I'M GOING TO DO IT 
        # ON THE TRAINING SET.
        ds = train_dss[0]
        ds.consecutive_frames = 1
        ds.skips = 1
        ds.frames = ds.frames[0:4]
        ds.angles = ds.angles[0:4]
        dl = DataLoader(ds, batch_size=1)
        visualization.trigger_activation_maps(dl)
        visualization.save_activation_maps()


    if save_history:
        history.save_training_info()
    
    
    if test_model:
        #load test data here
        pass
    
    
    if save_model:
        history.save_model()

if __name__ == "__main__":
    main()