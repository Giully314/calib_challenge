import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time 
import utils as ut
import pickle
import os
from matplotlib import pyplot as plt

#TODO Function that train and test the model to automate the process as mush as i can.

#TODO Add test evaluation values. (Inherent of what the model needs to do.)
class History:
    def __init__(self, dir: str, model: nn.Module, opt: torch.optim, loss_func: nn.Module, scheduler: torch.optim.lr_scheduler, 
                    batch_size: int):
        self.dir = dir
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.history = {"train_loss": [], "valid_loss": [], "total_time": 0}

        ut.create_dir(self.dir)

    def save(self):
        #TODO build one string and then write, for optimizing access to disk.
        file_txt = os.path.join(self.dir, "history.txt")
        with open(file_txt, "w") as f:
            # for key, value in self.__dict__.items():
            #     if key == "filename" or key == "history" or key == "scheduler" or key == batch_size:
            #         continue
            #     if value is not None:
            #         f.write(f"{repr(value)}\n\n")
            
            s = f"{repr(self.model)}\n\n"
            s += f"{repr(self.opt)}\n\n"
            s += f"{repr(self.loss_func)}\n\n"

            if self.scheduler is not None:
                s += f"{str(type(self.scheduler))}"
                s += f"\n{self.scheduler}\n\n"

            f.write(s)

            f.write("Training:\n")
            for i, (train_loss, valid_loss) in enumerate(zip(self.history["train_loss"], self.history["valid_loss"])):
                f.write(f"Epoch {i}\nTrain loss: {train_loss}\nValid loss: {valid_loss}\n\n")

            total_time = self.history["total_time"]
            f.write(f"\nTotal time: { total_time }")

        h_file = os.path.join(self.dir, "history.pkl")
        with open(h_file, "wb") as f:
            pickle.dump(self.history, f)


    def load(self):
        h_file = os.path.join(self.dir, "history.pkl")
        with open(h_file, "rb") as f:
            self.history = pickle.load(f)
    
    
    def save_png(self):
        #TODO create a new figure and then plot.
        train_loss = self.history["train_loss"]
        val_loss = self.history["valid_loss"]
        epochs = [i for i in range(len(train_loss))]
        fig = plt.figure(figsize=(10, 8), dpi=80)
        plt.plot(epochs, train_loss, "b-", label="TrainLoss")
        plt.plot(epochs, val_loss, "g-", label="ValidLoss")
        plt.legend(loc="center right", fontsize=12) 
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss)) + 0.01])
        h_file = os.path.join(self.dir, "history.png")
        plt.savefig(h_file, transparent=False)


    def __getitem__(self, idx):
        if idx != "train_loss" and idx != "valid_loss" and idx != "total_time":
            raise KeyError
        return self.history[idx]

    def __setitem__(self, idx, value):
        if idx != "total_time":
            raise KeyError
        self.history[idx] = value
        
    def __str__(self):
        return "TODO History str" 



def loss_batch(model: nn.Module, loss_func: nn.Module, xb: torch.Tensor, yb: torch.Tensor, opt: torch.optim = None):
    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)


def fit(epochs: int, history: History, train_dls: DataLoader, valid_dls: DataLoader, dev: torch.device,
        verbose=True) -> History:
    """
    Return an History object.
    """
    model = history.model
    opt = history.opt
    loss_func = history.loss_func
    scheduler = history.scheduler

    total_time = 0
    for epoch in range(epochs):
        since = time.time()
        
        if verbose:
            print(f"Start epoch #{epoch}...")
        
        model.train()
        train_loss = 0       
        for train_dl in train_dls:
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt) for xb, yb in train_dl])
            train_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)
            
        history["train_loss"].append(train_loss)
        
        if scheduler is not None:
            scheduler.step()

        if valid_dls is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():        
                for valid_dl in valid_dls:  
                    losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl])
                    val_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)

            history["valid_loss"].append(val_loss)

        time_elapsed = time.time() - since
        total_time += time_elapsed

        if verbose:
            print(f"Train_loss: {train_loss}")
            print(f"Val_loss: {val_loss}")
            print(f'Epoch complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s\n')

    history["total_time"] = total_time
    history.save()
    history.save_png()

    if verbose:
        print(f"Total time for training {(total_time//60):.0f}m {(total_time % 60):.0f}s")
    
    return history



