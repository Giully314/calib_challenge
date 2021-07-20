import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time 
from models import NvidiaModel

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

