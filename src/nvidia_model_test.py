from models import BlockArgs, RMSELoss, AugmentedNvidiaModel, NvidiaModel
from torch.utils.data import DataLoader
import torch
from datasets import get_frame_ds, get_range_frame_ds
import train_nvidia_model as tnm
import os 
import numpy as np
from torch import nn
import random 
import hydra
from omegaconf import DictConfig, OmegaConf

#TODO add verbose mode with timing and ecc...

@hydra.main(config_path="config", config_name="nvidia_train.yaml")
def main(cfg: DictConfig):
    seed = 1729
    def reset_rand_seed(): 
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    reset_rand_seed()
    #torch.use_deterministic_algorithms(True)

    args = cfg["train"]

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_size = tuple(args.frame_size)

    data_dir = args.data
    videos = args.videos
    train_dir = os.path.join(data_dir, args.train_dir)
    train_dirs = [os.path.join(train_dir, d) for d in args.train_data]
    train_dirs = [os.path.join(d, str(video)) for d in train_dirs for video in videos]
    # train_angles_files = [os.path.join(td, "angles.txt") for td in train_dirs]
    
    valid_dir = os.path.join(data_dir, args.valid_data)
    valid_dirs = [os.path.join(valid_dir, str(video)) for video in videos]
    # valid_angles_files = [os.path.join(vd, "angles.txt") for vd in valid_dirs]

    test_dir = os.path.join(data_dir, args.test_data)
    test_dirs = [os.path.join(test_dir, str(video)) for video in videos]
    # test_angles_files = [os.path.join(td, "angles.txt") for td in test_dirs]
    
    if args.dataset == "FrameDataset":
        train_datasets = [get_frame_ds(frame_dir) for frame_dir in train_dirs]
        valid_datasets = [get_frame_ds(frame_dir) for frame_dir in valid_dirs]
        test_datasets = [get_frame_ds(frame_dir) for frame_dir in test_dirs]
    elif args.dataset == "RangeFrameDataset":
        train_datasets = [get_range_frame_ds(frame_dir) for frame_dir in train_dirs]
        valid_datasets = [get_range_frame_ds(frame_dir) for frame_dir in valid_dirs]
        test_datasets = [get_frame_ds(frame_dir) for frame_dir in test_dirs]
    else:
        print("Attention no dataset provided.")


    batch_size = args.batch_size
    train_workers = args.train_workers
    valid_workers = args.valid_workers
    test_workers = args.test_workers
    shuffle = args.shuffle
    persistent_workers = args.persistent_workers
    train_dataloaders = [
        DataLoader(train_ds, batch_size,
        num_workers=train_workers, shuffle=shuffle, pin_memory=True, persistent_workers=persistent_workers) 
        for train_ds in train_datasets]

    valid_dataloaders = [
        DataLoader(valid_ds, batch_size, 
        num_workers=valid_workers, shuffle=False, pin_memory=True, persistent_workers=persistent_workers) 
        for valid_ds in valid_datasets]

    test_dataloaders = [
        DataLoader(test_ds, 1, 
        num_workers=test_workers, shuffle=False, pin_memory=True) 
        for test_ds in test_datasets]

    

    if args.model == "nvidia":
        model = NvidiaModel(img_size)
        model.to(dev)
    elif args.model == "augmented_nvidia":
        blocks_args = None
        if args.blocks_args is not None:
            blocks_args = []
            for block_arg in args.blocks_args:
                blocks_args.append(BlockArgs(*block_arg))

        model = AugmentedNvidiaModel(img_size, blocks_args, args.linear_layers)
        model.to(dev)
    
    print(model.cnn_shape_out)
    
    #TODO add support for passing other parameters to the optimizer
    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)


    if args.loss == "mse":
        loss = nn.MSELoss()
    elif args.loss == "rmse":
        loss = RMSELoss()


    history = tnm.History(args.history_out, videos, model, opt, loss, None, batch_size)
    epochs = args.epochs
    tnm.fit(epochs, history, train_dataloaders, valid_dataloaders, dev, verbose=args.verbose)

    if args.save_history:
        history.save_training_info()
        history.test_model(test_dataloaders, dev)   
    if args.save_model:
        history.save_model()

if __name__ == "__main__":
    main()