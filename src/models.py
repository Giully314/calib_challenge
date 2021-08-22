import torch
from torch import nn
from torch.nn import functional as F
import functools 
import operator
import collections


from utils import (
    conv1x1, 
    conv2d,
    round_filters,
    round_repeats,
)



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, output, target):
        return torch.sqrt(F.mse_loss(output, target) + 1e-6)



BlockArgs = collections.namedtuple("BlockArgs", [
            "num_repeat", "in_channels", "out_channels", "kernel_size", "stride",
])


GlobalParams = collections.namedtuple("GlobalParams", [
    "width_coeff", "depth_coeff", "image_size",
    "conv_layers", #first layers of the conv 
    "final_pooling_layer",
    "num_recurrent_layers", "hidden_dim", "rec_dropout", #recurrent networks for temporal dependencies 
    "dropout",
    "fc_layers", #fully connected layers (tuple of ints)
])


GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)



class ConvBlock(nn.Module):
    def __init__(self, block_args: BlockArgs):
        super(ConvBlock, self).__init__()
        self.block_args = block_args

        
        self.conv1 = conv2d(self.block_args.in_channels, self.block_args.out_channels, self.block_args.kernel_size, self.block_args.stride)
        self.norm1 = nn.BatchNorm2d(self.block_args.out_channels)

        self.conv2 = conv2d(self.block_args.out_channels, self.block_args.out_channels, self.block_args.kernel_size)
        self.norm2 = nn.BatchNorm2d(self.block_args.out_channels)
        
        
        self.activation_fun = nn.ELU(inplace=True)
        self.proj = None 
        if self.block_args.stride != 1 or (self.block_args.in_channels != self.block_args.out_channels):
            self.proj = nn.Sequential(
                conv1x1(self.block_args.in_channels, self.block_args.out_channels, self.block_args.stride),
                nn.BatchNorm2d(self.block_args.out_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation_fun(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.proj is not None:
            identity = self.proj(identity)
            
        out += identity
        out = self.activation_fun(out)
        
        return out



class NvidiaModel(nn.Module):
    def __init__(self, img_size: list[int]):
        super(NvidiaModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            
            nn.Conv2d(3, 24, 5, 2),
            nn.ELU(),

            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),

            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),

            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),

            nn.AvgPool2d(2),

            nn.Flatten(1)
        )
        h, w = img_size
        out = self.cnn(torch.zeros(1, 3, h, w))
        self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))
        
        self.dropout = nn.Dropout(0.0)

        self.linear = nn.Sequential(
            nn.Linear(self.cnn_shape_out, 100),
            nn.ELU(), 
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(), 
            nn.Linear(10, 2),
        )

    def forward(self, X):
        out = self.cnn(X)
        out = self.dropout(out)
        out = self.linear(out)
        return out


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


#Basic architecture for testing the skeleton of the training script
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),

            nn.AvgPool2d(2),

            nn.Flatten(1)
        )

        out = self.cnn(torch.zeros(1, 1, 28, 28))
        self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))
        
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Sequential(
            nn.Linear(self.cnn_shape_out, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, X):
        out = self.cnn(X)
        out = self.dropout(out)
        out = self.linear(out)
        return out



class AugmentedNvidiaModel(nn.Module):
    def __init__(self, img_size: list[int], blocks_args: list[BlockArgs], linear_args: list[tuple[int, int]]):
        super(AugmentedNvidiaModel, self).__init__()

        #Nvidia basic as starting model.
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            
            nn.Conv2d(3, 24, 5, 2, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),

            nn.Conv2d(24, 36, 5, 2, bias=False),
            nn.BatchNorm2d(36),
            nn.ELU(),

            nn.Conv2d(36, 48, 5, 2, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(),

            nn.Conv2d(48, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(64, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Flatten(1)
        )   
        
        h, w = img_size
        out = self.extract_features(torch.zeros(1, 3, h, w))
        self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))
        
        # self.dropout = nn.Dropout(0.5)

        self.linear = nn.Sequential(
            nn.Linear(self.cnn_shape_out, 100),
            nn.ELU(), 
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(), 
            nn.Linear(10, 2),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.cnn(X)

        # out = self.dropout(out)
        out = self.linear(out)
        return out