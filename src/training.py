import torch
import numpy as np
import time 
from ray import tune
from utils import RMSELoss

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dls, valid_dls, dev, scheduler = None) -> dict:
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
        for train_dl in train_dls:
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt) for xb, yb in train_dl])
            train_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)
                                        
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():      
            for valid_dl in valid_dls: 
                model.zero_hidden()   
                losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl])
                val_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)


        if scheduler is not None:
            scheduler.step(val_loss)


        history["val_loss"].append(val_loss)

        time_elapsed = time.time() - since
        total_time += time_elapsed
        print(f"Train_loss: {train_loss}")
        print(f"Val_loss: {val_loss}")
        print(f'Epoch complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s\n')

    print(f"Total time for training {(total_time//60):.0f}m {(total_time % 60):.0f}s")
    return history  



def fit_hyper_params(config, data_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Endurance(c1=config['c1'], l1=config['l1'], l2=config['l2'])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    loss_func = ut. 

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    train_angles = os.path.join(train_dir, "angles.txt")
    valid_angles = os.path.join(valid_dir, "angles.txt")
    train_ds = FrameDataset(train_dir, train_angles)
    valid_ds = FrameDataset(valid_dir, valid_angles)

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=True, pin_memory=True)

    epochs = 15
    for epoch in range(epochs):   
        model.train()
        train_loss = 0
        
        losses, nums = zip(*[loss_batch(model, loss_func, xb.to(device), yb.to(device), opt) for xb, yb in train_dl])
        train_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():          
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in valid_dl])
            val_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), opt.state_dict()), path)
        
        tune.report(val_loss=val_loss)

    print("Finished training")