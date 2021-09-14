import torch
from torch import nn
# from torch.utils.data import DataLoader, Data

import numpy as np

from dataclasses import dataclass, field 

import os
from datasets import ConsecutiveFramesDataset

import utils as ut

import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

#TODO: plot angles of training video vs predicted angles on training video
@dataclass
class History:
    """
    Class that stores information about: model, loss, optimizer, scheduler, batch_size, 
    what videos used for training, valid video, test video.
    All the graphs about the training process with training loss and valid loss compared or not.
    Test informations with the result and the respectives graphs.
    """

    dir: str
    train_videos: list[str]
    valid_videos: str
    model: torch.nn.Module
    opt: torch.optim
    loss_func: nn.Module
    scheduler: torch.optim.lr_scheduler
    batch_size: int
    active: bool = True

  
    def __post_init__(self):
        if not self.active:
            return

        self.history = {"train_loss": [], "valid_loss": [], "total_time": 0.0}
        self.dir = os.path.join(self.dir, "history")
        ut.create_dir(self.dir)


    def save_training_info(self):
        if not self.active:
            return
    
        file_txt = os.path.join(self.dir, "history.txt")

        valid = len(self.history["valid_loss"]) > 0
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        with open(file_txt, "w") as f:
            s = f"{repr(self.model)}\n\n"
            s += f"Model total params: {total_params}\n\n"
            s += f"{repr(self.opt)}\n\n"
            s += f"{repr(self.loss_func)}\n\n"

            if self.scheduler is not None:
                s += f"{str(type(self.scheduler))}"
                s += f"\n{repr(self.scheduler)}\n\n"

            s += f"Batch size: {self.batch_size}\n\n"

            s += f"Training videos {self.train_videos}\n"
            if self.valid_videos is not None:
                s += f"Valid videos {self.valid_videos}\n\n"

            s += "Training:\n"
            if valid:
                for i, (train_loss, valid_loss) in enumerate(zip(self.history["train_loss"], self.history["valid_loss"])):
                    s += f"Epoch {i}\nTrain loss: {train_loss}\nValid loss: {valid_loss}\n\n"
            else:
                for i, train_loss  in enumerate(self.history["train_loss"]):
                    s += f"Epoch {i}\nTrain loss: {train_loss}\n\n"

            total_time = self.history["total_time"]
            s += f"\nTotal time: { total_time }"

            f.write(s)

        #The training process used a validation set so it's safe to compare the training loss with the validation loss.
        if valid:
            self.save_training_valid_curve() 

        #Save only the training curve.
        self.save_training_curve()

    def save_training_curve(self):
        if not self.active:
            return

        train_loss = self.history["train_loss"]
        epochs = [i for i in range(len(train_loss))]

        fig, ax = plt.subplots(figsize=(14, 12), dpi=160)
        ax.plot(epochs, train_loss, "b-", label="TrainLoss")
        ax.set_ylim(0, min(train_loss) * 10)
        ax.legend(loc="center right", fontsize=12) 
        ax.set_xlabel("Epoch", fontsize=16)
        lr = self.opt.param_groups[0]["lr"]
        ax.set_ylabel(f"Train Loss with lr {lr} and min {min(train_loss)}", fontsize=16)
        
        h_file = os.path.join(self.dir, "training_curve.png")
        plt.savefig(h_file, dpi=160)
        plt.close(fig)
    
    
    def save_training_valid_curve(self):
        if not self.active:
            return

        train_loss = self.history["train_loss"]
        val_loss = self.history["valid_loss"]
        epochs = [i for i in range(len(train_loss))]

        fig, ax = plt.subplots(figsize=(14, 12), dpi=160)
        ax.plot(epochs, train_loss, "b-", label="TrainLoss")
        ax.plot(epochs, val_loss, "g-", label="ValidLoss")
        ax.legend(loc="center right", fontsize=12) 
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss))])
        
        h_file = os.path.join(self.dir, "training_valid_curve.png")      
        plt.savefig(h_file, dpi=160)
        plt.close(fig)


    def save_model(self) -> None:
        if not self.active:
            return

        path = os.path.join(self.dir, "model_state_dict.pt")
        ut.save_model(self.model, path)


    def __getitem__(self, idx):
        if not self.active:
            return

        if idx != "train_loss" and idx != "valid_loss" and idx != "total_time":
            raise KeyError
        return self.history[idx]

    def __setitem__(self, idx, value):
        if not self.active:
            return

        if idx != "total_time":
            raise KeyError
        self.history[idx] = value


    def __bool__(self):
        return self.active


@dataclass
class ActivationMapVisualization:
    """
    Visualize activation map, filters, gradient, saliency map ecc.
    The model should contains an attribute called cnn of type nn.Sequential(OrderedDict())    
    """
    dir: str
    model: nn.Module
    dev: torch.device
    active: bool = True
    

    def __post_init__(self):
        if not self.active: #the class is not active, so don't initialize the resources.
            return

        self.activation_map_hook = ut.ActivationMapHook()
        self.dir = os.path.join(self.dir, "activation_map_visual")
        ut.create_dir(self.dir)


    def register_activation_map(self, layer_name: str) -> None:
        """
        Works only for cnn.
        """
        i = ut.get_index_by_name(self.model.cnn, layer_name)
        self.model.cnn[i].register_forward_hook(self.activation_map_hook.get_activation(layer_name))


    def save_activation_maps(self):
        """
        Save all the activation maps. I only consider the first 3 conv layer. I fix the number of columns in the figure (3).
        """
        for layer_name, activations in self.activation_map_hook.activation.items():
            d = os.path.join(self.dir, "activation_" + str(layer_name))
            ut.create_dir(d)
            n_cols = 3
            n_rows = activations[0].shape[1] // n_cols

            #TODO: clear the figure instaed of creating a new one (maybe it's better in terms of performance)
            for i in range(len(activations)):
                fig, ax_array = plt.subplots(n_rows, n_cols, figsize=(14, 12), dpi=160)
                file = os.path.join(d, str(i) + ".png")

                act = activations[i].squeeze().cpu()
                for i in range(n_rows):
                    for j in range(n_cols):
                        ax_array[i, j].imshow(act[i * n_cols + j], cmap='gray')
                
                plt.savefig(file, dpi=160)
                plt.close(fig)


    def trigger_activation_maps(self, dss: list[ConsecutiveFramesDataset], frames: list[int]):  
        """
        ds: the dataset.
        frames: which frames to use.
        """
        temp_cons_frames = dss[0].consecutive_frames
        temp_skips = dss[0].skips
        
        for ds in dss:
            ds.consecutive_frames = 1
            ds.skips = 1

        self.model.to(torch.device("cpu"))
        self.model.eval()
        
        for i, ds in enumerate(dss):
            for frame in frames[i]:
                x, y = ds[frame]
                self.model.extract_features(x.unsqueeze(0))

        for ds in dss:
            ds.consecutive_frames = temp_cons_frames
            ds.skips = temp_skips

    
    def __bool__(self):
        return self.active


@dataclass
class GradientFlowVisualization:
    """
    Class that register the gradient flow every t epochs. This class istance can be used only one time;
    create an istance for each different use.
    """
    
    dir: str
    epochs: int = 1 #how often register the gradient flow.
    active: bool = True #Activate this class.

    def __post_init__(self):
        if not self.active: #the class is not active, so don't initialize the resources.
            return

        self.dir = os.path.join(self.dir, "gradient_flow_visual")
        ut.create_dir(self.dir)
        self.fig = None
        self.ax = None
        self.count = 0
        #self.fig, self.ax = plt.subplots(figsize=(8, 6))


    def register_gradient_flow(self, named_parameters):    
        if not self.active:
            return
        
        if self.count % self.epochs == 0:
            ave_grads = []
            max_grads= []
            layers = []
            #This loop is memory intensive because we need to copy the data into the cpu before to use them.
            for n, p in named_parameters:
                if(p.requires_grad) and ("bias" not in n):
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())

            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 6))
                self.ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
                plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
                self.ax.set_xlim(left=0, right=len(ave_grads))
                self.ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
                self.ax.set_xlabel("Layers")
                self.ax.set_ylabel("average gradient")
                self.ax.set_title("Gradient flow")
                self.ax.grid(True)
                # self.ax.legend([Line2D([0], [0], color="c", lw=4),
                #         Line2D([0], [0], color="b", lw=4),
                #         Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
            
            #maybe split the max_grads and ave_grads into 2 differents plots.
            self.ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
            self.ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
            # self.ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
            # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
            # self.ax.set_xlim(left=0, right=len(ave_grads))
            # self.ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
            # self.ax.set_xlabel("Layers")
            # self.ax.set_ylabel("average gradient")
            # self.ax.set_title("Gradient flow")
            # self.ax.grid(True)
            # self.ax.legend([Line2D([0], [0], color="c", lw=4),
            #             Line2D([0], [0], color="b", lw=4),
            #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        self.count += 1


    def save_gradient_flow(self):
        if not self.active:
            return

        file = os.path.join(self.dir, "gradient_flow.png")
        # plt.legend(loc='upper right')
        plt.savefig(file)
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.active = False
    
    
    def __bool__(self):
        return self.active


class TestModel:
    def __init__(self, test_output: str, gt_angles_path: str, ds: ConsecutiveFramesDataset, model: nn.Module):
        self.test_output = os.path.join(test_output, "results")
        ut.create_dir(self.test_output)
        
        self.gt_angles_path = gt_angles_path
        self.predictions = os.path.join(self.test_output, "predictions.txt")
        self.ds = ds
        self.model = model 

    def test(self):
        self.eval_save_predictions()
        r = self.__eval_single_test()
        self.plot_results(r)
        
    #TODO: consider only the last prediction (maybe it's the most accurate) or consider all the predictions?
    def eval_save_predictions(self):
        self.model.to(torch.device("cpu"))
        self.model.eval()

        i = 0
        preds = []
        while i < len(self.ds):
            frames, _ = self.ds[i]
            preds.append(self.model(frames.unsqueeze(0)).squeeze(0).detach().numpy().reshape(-1, 2))
            i += self.ds.consecutive_frames
        
        preds = np.concatenate(preds, axis=0)
        np.savetxt(self.predictions, preds)


    def __eval_single_test(self):
        def get_mse(gt, test):
            test = np.nan_to_num(test)
            return np.mean(np.nanmean((gt - test)**2, axis=0))

        zero_mses = []
        mses = []

        gt = np.loadtxt(self.gt_angles_path)
        zero_mses.append(get_mse(gt, np.zeros_like(gt)))

        test = np.loadtxt(self.predictions)
        mses.append(get_mse(gt, test))

        percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
        # print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
        return percent_err_vs_all_zeros


    def plot_results(self, result_perc: int = None):
        fig, axs = plt.subplots(2, 1, figsize=(14, 12), dpi=160)
        fig.suptitle(f"GT vs Predictions: error score {result_perc}%")

        gt = np.loadtxt(self.gt_angles_path)
        preds = np.loadtxt(self.predictions)
        num_angles = [i for i in range(gt.shape[0])]    

        axs[0].set_title("Pitch")
        axs[0].plot(num_angles, gt[:, 0], linewidth=2, color='green', label="gt")
        axs[0].plot(num_angles, preds[:, 0], linewidth=2, color='blue', label="pred")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Angles in radians")
        axs[0].legend("best")
        axs[0].set(xlim=(0, len(num_angles) + 1), ylim=(0, 0.1))

        axs[1].set_title("Yaw")
        axs[1].plot(num_angles, gt[:, 1], linewidth=2, color='green', label="gt")
        axs[1].plot(num_angles, preds[:, 1], linewidth=2, color='blue', label="pred")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Angles in radians")
        axs[1].legend("best")
        axs[1].set(xlim=(0, len(num_angles) + 1), ylim=(0, 0.1))

        plt.savefig(os.path.join(self.test_output, "results.png"), dpi=100)
        plt.close(fig)


