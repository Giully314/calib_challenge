import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time 
from models import NvidiaModel
import pickle
import os
from matplotlib import pyplot as plt

class History:
    def __init__(self, filename: str, model: nn.Module, opt: torch.optim, loss_func: nn.Module, batch_size: int):
        self.filename = filename
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.history = {"train_loss": [], "valid_loss": []}

    def save(self):
        with open(self.filename, "w") as f:
            for key, value in self.__dict__.items():
                if key == "filename" or key == "history":
                    continue
                if value is not None:
                    f.write(f"{repr(value)}\n\n")

            f.write("Training:\n")
            for i, (train_loss, valid_loss) in enumerate(zip(self.history["train_loss"], self.history["valid_loss"])):
                f.write(f"Epoch {i}\nTrain loss: {train_loss}\nValid loss: {valid_loss}\n\n")

        h_file = os.path.splitext(self.filename)[0] + ".pkl"
        with open(h_file, "wb") as f:
            pickle.dump(self.history, f)


    def load(self):
        h_file = os.path.splitext(self.filename)[0] + ".pkl"
        with open(h_file, "rb") as f:
            self.history = pickle.load(f)
    
    
    def save_png(self):
        #TODO create a new figure and then plot.
        train_loss = self.history["train_loss"]
        val_loss = self.history["valid_loss"]
        epochs = [i for i in range(len(train_loss))]
        plt.plot(epochs, train_loss, "b-", label="TrainLoss")
        plt.plot(epochs, val_loss, "g-", label="ValidLoss")
        plt.legend(loc="center right", fontsize=12) 
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss)) + 0.1])
        h_file = os.path.splitext(self.filename)[0] + ".png"
        plt.savefig(h_file)

    def __getitem__(self, idx):
        if idx != "train_loss" and idx != "valid_loss":
            raise KeyError
        return self.history[idx]

    def __str__(self):
        return "TODO History str" 



def loss_batch(model: nn.Module, loss_func: nn.Module, xb: torch.Tensor, yb: torch.Tensor, opt: torch.optim = None):
    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)


def fit_nvidia_model(epochs: int, model: NvidiaModel, loss_func: nn.Module, opt: torch.optim, 
        scheduler: torch.optim.lr_scheduler.MultiStepLR, train_dl: DataLoader, valid_dl: DataLoader, dev: torch.device) -> dict:
    """
    Return a dict with 2 keys: train_loss and val_loss.
    Each element of the list correspond to the loss of the respective epoch idx.
    """

    history = {"train_loss": [], "val_loss": []}
    total_time = 0
    for epoch in range(epochs):
        since = time.time()
        print(f"Start epoch #{epoch}...")
        
        model.train()
        train_loss = 0
        
        losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt) for xb, yb in train_dl])
        train_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)
            
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():          
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl])
            val_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if scheduler is not None:
            scheduler.step()


        history["val_loss"].append(val_loss)

        time_elapsed = time.time() - since
        total_time += time_elapsed
        print(f"Train_loss: {train_loss}")
        print(f"Val_loss: {val_loss}")
        print(f'Epoch complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s\n')

    print(f"Total time for training {(total_time//60):.0f}m {(total_time % 60):.0f}s")
    return history

