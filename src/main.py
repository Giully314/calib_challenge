from models import CalibModel, CalibParams, RMSELoss
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
from informations import History, GradientFlowVisualization, ActivationMapVisualization

#TODO ADD LOGGING

@hydra.main(config_path="config")
def main(cfg: DictConfig):
    args = cfg["training"]["train"]
    
    img_size = tuple(args.frame_size)

    timer = Timer()

    if args.deterministic:
        seed = 1729
        def reset_rand_seed(): 
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        reset_rand_seed()
        #torch.use_deterministic_algorithms(True)

    if not args.debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(enabled=False)


    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"DEVICE {dev}")


    train_videos = [os.path.join(args.data_dir, os.path.join(str(video), str(part) + ".pt")) 
                    for video, video_parts in zip(args.train_videos, args.videos_parts) for part in video_parts]
    train_angles = [os.path.join(args.data_dir, os.path.join(str(video), str(part) + ".txt")) 
                            for video, video_parts in zip(args.train_videos, args.videos_parts) for part in video_parts]
    

    #TODO implement a config file for transformations. 

    print("Loading datasets...", end=" ")
    timer.start()
    train_dss = [get_consecutive_frames_ds(video_path, angles_path, args.consecutive_frames, args.skips) 
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
        DataLoader(train_ds, args.batch_size,
        num_workers=args.train_workers, shuffle=args.shuffle, pin_memory=True, persistent_workers=args.persistent_workers) 
        for train_ds in train_dss]


    valid_dls = None
    if args.valid_model:
        pass
    
    calib_params = CalibParams(img_size, args.consecutive_frames, args.lstm_hidden_size, args.lstm_num_layers, args.linear_layers)
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
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    if args.loss == "mse":
        loss = nn.MSELoss()
    elif args.loss == "rmse":
        loss = RMSELoss()


    scheduler =  torch.optim.lr_scheduler.MultiStepLR(opt, args.scheduler_epochs, gamma=args.scheduler_gamma)

    history = History(args.training_info_dir, args.train_videos, args.valid_video, 
                        model, opt, loss, scheduler, args.batch_size, args.history_active)
    
    grad_flow = GradientFlowVisualization(args.training_info_dir, args.grad_flow_epochs, args.grad_flow_active)
    
    activation_map = ActivationMapVisualization(args.training_info_dir, model, dev, args.activation_map_active)
    
    trn.fit(args.epochs, history, grad_flow, train_dls, valid_dls, dev, verbose=args.verbose)

    if args.history_save_train_info:
        history.save_training_info()
    
    if args.history_save_model:
        history.save_model()

    if args.test_model:
        #load test data here
        pass
    

    if activation_map:
        for layer_name in args.activation_map_layers:
            activation_map.register_activation_map(layer_name)

        #NOTE: TECHNICALLY THE VISUALIZATION OF THE ACTIVATION MAPS SHOULD BE DONE ON VALID/TEST SET. FOR NOW I'M GOING TO DO IT 
        # ON THE TRAINING SET.
        act_map_dss = [train_dss[i] for i in args.activation_map_dss]
        activation_map.trigger_activation_maps(act_map_dss, args.activation_map_frames)
        activation_map.save_activation_maps()


if __name__ == "__main__":
    main()