import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time 

from informations import History, GradientFlowVisualization

#TODO Function that train and test the model to automate the process as mush as i can.


def loss_batch(model: nn.Module, loss_func: nn.Module, xb: torch.Tensor, yb: torch.Tensor, opt: torch.optim = None, grad_visual: GradientFlowVisualization = None):
    y_pred = model(xb)
    loss = loss_func(y_pred, yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        
        if grad_visual:
            grad_visual.register_gradient_flow(model.named_parameters())
        
        opt.step()

    return loss.item(), len(xb)


#TODO: currently the to device operation is done async. Check if there are any speed up.
def fit(epochs: int, history: History, grad_flow: GradientFlowVisualization, train_dls: list[DataLoader], valid_dls: list[DataLoader], dev: torch.device,
        verbose: bool = True) -> History:
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
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True), opt, grad_flow) 
                                for xb, yb in train_dl])
            train_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)
        
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        info = f"Train_loss: {train_loss}\n"
    

        if valid_dls is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():        
                for valid_dl in valid_dls:  
                    losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl])
                    val_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)

            history["valid_loss"].append(val_loss)
            info += f"Valid_loss: {val_loss}\n" 

        time_elapsed = time.time() - since
        total_time += time_elapsed

        if verbose:
            info += f"Epoch complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s {((time_elapsed % 1) * 1000):.0f}ms\n"
            print(info) 

    history["total_time"] = total_time

    if grad_flow is not None:
        print("Saving gradient flow....")
        grad_flow.save_gradient_flow()

    if verbose:
        print(f"Total time for training {(total_time//60):.0f}m {(total_time % 60):.0f}s")
    
    return history


