import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from dataclasses import dataclass, field 

import os

import utils as ut

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    valid_video: str
    model: torch.nn.Module
    opt: torch.optim
    loss_func: nn.Module
    scheduler: torch.optim.lr_scheduler
    batch_size: int
    history: dict = field(init=False)

  
    def __post_init__(self):
        self.history = {"train_loss": [], "valid_loss": [], "total_time": 0.0}
        self.dir = os.path.join(self.dir, "history")
        ut.create_dir(self.dir)


    def save_training_info(self):
        file_txt = os.path.join(self.dir, "history.txt")

        valid = len(self.history["valid_loss"]) > 0

        with open(file_txt, "w") as f:
            s = f"{repr(self.model)}\n\n"
            s += f"{repr(self.opt)}\n\n"
            s += f"{repr(self.loss_func)}\n\n"

            if self.scheduler is not None:
                s += f"{str(type(self.scheduler))}"
                s += f"\n{self.scheduler}\n\n"

            s += f"Training videos {self.train_videos}\n"
            if self.valid_video is not None:
                s += f"Valid video {self.valid_video}\n\n"

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
        train_loss = self.history["train_loss"]
        epochs = [i for i in range(len(train_loss))]

        fig = plt.figure(figsize=(18, 16), dpi=160)
        ax = fig.add_subplot()
        ax.plot(epochs, train_loss, "b-", label="TrainLoss")
        ax.legend(loc="center right", fontsize=12) 
        ax.set_xlabel("Epoch", fontsize=16)
        lr = self.opt.param_groups[0]["lr"]
        ax.set_ylabel(f"Train Loss with lr {lr} and min {min(train_loss)}", fontsize=16)
        h_file = os.path.join(self.dir, "training_curve.png")
        plt.savefig(h_file, transparent=False)
    
    
    def save_training_valid_curve(self):
        train_loss = self.history["train_loss"]
        val_loss = self.history["valid_loss"]
        epochs = [i for i in range(len(train_loss))]

        fig = plt.figure(figsize=(18, 16), dpi=160)
        ax = fig.add_subplot()
        ax.plot(epochs, train_loss, "b-", label="TrainLoss")
        ax.plot(epochs, val_loss, "g-", label="ValidLoss")
        ax.legend(loc="center right", fontsize=12) 
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.axis([0, len(epochs)+1, 0, max(max(val_loss), max(train_loss)) + 0.01])
        h_file = os.path.join(self.dir, "training_valid_curve.png")
        plt.savefig(h_file, transparent=False)


    def save_model(self) -> None:
        path = os.path.join(self.dir, "model_state_dict.pt")
        ut.save_model(self.model, path)


    def __getitem__(self, idx):
        if idx != "train_loss" and idx != "valid_loss" and idx != "total_time":
            raise KeyError
        return self.history[idx]

    def __setitem__(self, idx, value):
        if idx != "total_time":
            raise KeyError
        self.history[idx] = value



@dataclass
class ActivationMapVisualization:
    """
    Visualize activation map, filters, gradient, saliency map ecc.
    The model should contains an attribute called cnn of type nn.Sequential(OrderedDict())    
    """
    dir: str
    model: nn.Module
    dev: torch.device
    verbose: bool = False
    

    def __post_init__(self):
        self.activation_map_hook = ut.ActivationMapHook()
        self.dir = os.path.join(self.dir, "activation_map_visual")
        ut.create_dir(self.dir)


    def register_activation_map(self, layer_name: str) -> None:
        """
        Works only for cnn.
        """
        if self.verbose:
            print(f"Register activation map for {layer_name}")
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
            
            if self.verbose:
                print(f"Save activation map for {layer_name}")
            for i in range(len(activations)):
                file = os.path.join(d, str(i) + ".png")

                if self.verbose:
                    print(f"Save file {file}")

                act = activations[i].squeeze().cpu()
                fig, ax_array = plt.subplots(n_rows, n_cols, figsize=(20, 18), dpi=160)
                for i in range(n_rows):
                    for j in range(n_cols):
                        ax_array[i, j].imshow(act[i * n_cols + j], cmap='gray')
                plt.savefig(file)
                plt.close(fig)

            if self.verbose:
                print()

    def trigger_activation_maps(self, dl: DataLoader):  
        """
        Trigger the registered activation maps.
        DATALOADER SHOULD HAVE BATCH_SIZE = 1.
        """
        if self.verbose:
            print("Trigger activation maps")
        self.model.eval()
        for x, y in dl:
            self.model(x.to(self.dev))

@dataclass
class GradientFlowVisualization:
    """
    USE THIS CLASS ONLY IF NO OTHER PYPLOT FIGURE IS USED INSIDE THE TRAINING LOOP 
    """
    
    dir: str

    def __post_init__(self):
        self.dir = os.path.join(self.dir, "gradient_flow_visual")
        ut.create_dir(self.dir)
        self.fig = None

    def register_gradient_flow(self, named_parameters):
        def plot_grad_flow(named_parameters):
            ave_grads = []
            layers = []
            for n, p in named_parameters:
                if(p.requires_grad) and ("bias" not in n):
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
            plt.plot(ave_grads, alpha=0.3, color="b")
            plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
            plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(xmin=0, xmax=len(ave_grads))
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            
        if self.fig is None:
            self.fig = plt.figure(figsize=(20, 18), dpi=160)
        
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


    def save_gradient_flow(self):
        file = os.path.join(self.dir, "gradient_flow.png")
        plt.savefig(file)
        plt.close(self.fig)
        self.fig = None


class TestModel:
    pass