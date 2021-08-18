from models import RMSELoss
from torch.utils.data import Dataset, DataLoader
import utils as ut
import torch
from datasets import FrameDataset
import train_nvidia_model as tnm
import os 
import numpy as np
import random 
import matplotlib.pyplot as plt
from torch import nn
import functools
import operator
import shutil

def main():
    seed = 1729
    def reset_rand_seed(): 
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    reset_rand_seed()
    #torch.use_deterministic_algorithms(True)


    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #if a conv layer is followed by a batchnorm, don't use bias (it cancel the effect so it's useless)
    class NvidiaModel(nn.Module):
        def __init__(self, img_size: list[int]):
            super(NvidiaModel, self).__init__()

            self.cnn = nn.Sequential(
                nn.BatchNorm2d(3),
                
                nn.Conv2d(3, 24, 5, 2),
                # nn.BatchNorm2d(24),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),

                nn.Conv2d(24, 36, 5, 2),
                # nn.BatchNorm2d(36),
                nn.ReLU(),

                nn.Conv2d(36, 48, 5, 2),
                # nn.BatchNorm2d(48),
                nn.ReLU(),

                nn.Conv2d(48, 64, 3, 1),
                # nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, 1),
                # nn.BatchNorm2d(64),
                nn.ReLU(),

                # nn.Conv2d(64, 128, 3, 2),
                # # nn.BatchNorm2d(128),
                # nn.ReLU(),

                # nn.Conv2d(128, 256, 3, 2),
                # nn.BatchNorm2d(256),
                # nn.ReLU(),


                # nn.AdaptiveAvgPool2d(1),
                # nn.AvgPool2d(2),

                nn.Flatten(1)
            )
            h, w = img_size
            out = self.cnn(torch.zeros(1, 3, h, w))
            self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))
            
            #self.dropout = nn.Dropout(0.5)

            self.linear = nn.Sequential(
                nn.Linear(self.cnn_shape_out, 1024),
                nn.ReLU(), 
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(), 
                nn.Linear(16, 2),
            )

        def forward(self, X):
            out = self.cnn(X)
            #out = self.dropout(out)
            out = self.linear(out)
            return out


    data_dir = "data_h3_w3"
    videos = [2, 3, 4]
    train_dir = os.path.join(data_dir, "train")
    train_dir = os.path.join(train_dir, "cropped")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    train_dirs = [os.path.join(train_dir, str(video)) for video in videos]
    valid_dirs = [os.path.join(valid_dir, str(video)) for video in videos]
    test_dirs = [os.path.join(test_dir, str(video)) for video in videos]
    train_angles_files = [os.path.join(td, "angles.txt") for td in train_dirs]
    valid_angles_files = [os.path.join(vd, "angles.txt") for vd in valid_dirs]
    test_angles_files = [os.path.join(td, "angles.txt") for td in test_dirs]



    train_datasets = [FrameDataset(frame_dir, angles_file) for frame_dir, angles_file in zip(train_dirs, train_angles_files)]
    valid_datasets = [FrameDataset(frame_dir, angles_file) for frame_dir, angles_file in zip(valid_dirs, valid_angles_files)]
    test_datasets = [FrameDataset(frame_dir, angles_file) for frame_dir, angles_file in zip(test_dirs, test_angles_files)]



    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2**32
        worker_seed = seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # g = torch.Generator()
    # g.manual_seed(seed)

    # w = torch.Generator()
    # w.manual_seed(seed)

    # t = torch.Generator()
    # t.manual_seed(seed)

    batch_size = 32
    num_workers = 0
    shuffle = False
    persistent_workers = False
    train_dataloaders = [
        DataLoader(train_ds, batch_size,
        num_workers=num_workers, shuffle=shuffle, pin_memory=True, persistent_workers=persistent_workers) 
        for train_ds in train_datasets]

    valid_dataloaders = [
        DataLoader(valid_ds, batch_size, 
        num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=persistent_workers) 
        for valid_ds in valid_datasets]

    test_dataloaders = [
        DataLoader(test_ds, 1, 
        num_workers=num_workers, shuffle=False, pin_memory=True) 
        for test_ds in test_datasets]

    img_size = (120, 360)


    model = NvidiaModel(img_size)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-04)
    loss = RMSELoss()
    #shutil.rmtree("test_1")
    history = tnm.History("test_1", videos, model, opt, loss, None, batch_size)
    epochs = 30
    tnm.fit(epochs, history, train_dataloaders, valid_dataloaders, dev, verbose=True)


    history.save_training_info()
    history.test_model(test_dataloaders, dev)   

if __name__ == "__main__":
    main()