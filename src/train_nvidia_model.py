import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time 
import utils as ut
import pickle
import os
from matplotlib import pyplot as plt

#Note: this file is for the training of the intermediate model (nvidia) for estimate NaN values.


#TODO Function that train and test the model to automate the process as mush as i can.

#TODO Add test evaluation values. (Inherent of what the model needs to do.)
class History:
    def __init__(self, dir: str, train_videos: list[str], model: nn.Module, opt: torch.optim, loss_func: nn.Module, 
                    scheduler: torch.optim.lr_scheduler, batch_size: int):
        self.dir = dir
        self.train_videos = train_videos    
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.history = {"train_loss": [], "valid_loss": [], "total_time": 0.0}

        ut.create_dir(self.dir)

    def save_training_info(self):
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
        
        self._save_training_png()


    def load(self):
        h_file = os.path.join(self.dir, "history.pkl")
        with open(h_file, "rb") as f:
            self.history = pickle.load(f)
    
    
    def _save_training_png(self):
        #TODO create a new figure and then plot.
        train_loss = self.history["train_loss"]
        val_loss = self.history["valid_loss"]
        epochs = [i for i in range(len(train_loss))]

        fig = plt.figure(figsize=(16, 14), dpi=80)
        ax = fig.add_subplot()
        ax.plot(epochs, train_loss, "b-", label="TrainLoss")
        ax.plot(epochs, val_loss, "g-", label="ValidLoss")
        ax.legend(loc="center right", fontsize=12) 
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss)) + 0.01])
        h_file = os.path.join(self.dir, "history.png")
        plt.savefig(h_file, transparent=False)


    #test_model assume dataloaders to have batch_size = 1
    def test_model(self, test_dls: list[DataLoader], dev: torch.device) -> str:
        output_dir = os.path.join(self.dir, "results") 
        output_inferences = [os.path.join(output_dir, str(video) + ".txt") for video in self.train_videos]
        output_tests = [os.path.join(output_dir, str(video) + "_test.txt") for video in self.train_videos]
        output_result = os.path.join(output_dir, "result.txt")

        ut.create_dir(output_dir)
        #TODO compute only 1 time the test set.

        for test_dl, output_inference in zip(test_dls, output_inferences):
            ut.inference_and_save(self.model, test_dl, output_inference, dev) 
        
        for output_test, test_dl in zip(output_tests, test_dls):
            with open(output_test, "w") as f:
                for x, y in test_dl:
                    for i in range(y.shape[0]):
                        f.write(f"{y[i, 0].item()} {y[i, 1].item()}\n")

        mses = []
        zero_mses = []
        for output_test, output_inference in zip(output_tests, output_inferences):
            mse, zero_mse = ut.eval_angles(output_test, output_inference)
            mses.append(mse)
            zero_mses.append(zero_mse)

        percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)

        result_string = f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)'
        
        with open(output_result, "w") as f:
            f.write(result_string)

        self._save_test_png()

        return result_string


    def _save_test_png(self):
        results_dir = os.path.join(self.dir, "results") 
        inferred_angles_files = [os.path.join(results_dir, str(video) + ".txt") for video in self.train_videos]
        gt_angles_files = [os.path.join(results_dir, str(video) + "_test.txt") for video in self.train_videos]
        result = os.path.join(results_dir, "result.txt")

        inferred_angles = [np.loadtxt(inferred_angles_file) for inferred_angles_file in inferred_angles_files]
        gt_angles = [np.loadtxt(gt_angles_file) for gt_angles_file in gt_angles_files]

        with open(result, "r") as f:
            result_string = f.readline()

        for video, pred_angles, true_angles in zip(self.train_videos, inferred_angles, gt_angles):
            num_frames = len(pred_angles) #that's equal to len(gt_angles)
            frames = [i for i in range(num_frames)]
            fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 14), dpi=80)

            fig.suptitle(result_string, fontsize=14, fontweight='bold')

            ax1.plot(frames, pred_angles[:, 0], "b-", label="Inferred pitch")
            ax1.plot(frames, true_angles[:, 0], "g-", label="Gt pitch")
            ax1.legend(loc="center right", fontsize=12) 
            ax1.set_xlabel("Frame", fontsize=16)
            ax1.set_ylabel("Angles (rad)", fontsize=16)
            ax1.axis([0, num_frames+1, 0, max(max(pred_angles[:, 0]), max(true_angles[:, 0])) + 0.01])

            ax2.plot(frames, pred_angles[:, 1], "b-", label="Inferred yaw")
            ax2.plot(frames, true_angles[:, 1], "g-", label="Gt yaw")
            ax2.legend(loc="center right", fontsize=12) 
            ax2.set_xlabel("Frame", fontsize=16)
            ax2.set_ylabel("Angles (rad)", fontsize=16)
            ax2.axis([0, num_frames+1, 0, max(max(pred_angles[:, 1]), max(true_angles[:, 1])) + 0.01])

            h_file = os.path.join(results_dir, str(video) + ".png")
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
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return loss.item(), len(xb)

#TODO: currently the to device operation is done async. Check if there are any speed up.
#TODO: use the string info for history recap instaed of store sparse information.
def fit(epochs: int, history: History, train_dls: list[DataLoader], valid_dls: list[DataLoader], dev: torch.device,
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
        
        info = ""
        model.train()
        train_loss = 0       
        for train_dl in train_dls:
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev, non_blocking=False), yb.to(dev, non_blocking=False), opt) 
                                for xb, yb in train_dl])
            train_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)
            
        history["train_loss"].append(train_loss)
        info += f"Train_loss: {train_loss}\n"
        
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
            info += f"Valid_loss: {val_loss}\n" 

        time_elapsed = time.time() - since
        total_time += time_elapsed

        if verbose:
            info += f'Epoch complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s\n'
            print(info)

    history["total_time"] = total_time

    if verbose:
        print(f"Total time for training {(total_time//60):.0f}m {(total_time % 60):.0f}s")
    
    return history


