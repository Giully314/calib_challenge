import torch
from torch import nn
from torch.nn import functional as F
import functools 
import operator
import collections


from utils import (
    conv1x1, 
    conv2d,
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
        
        
        self.activation_fun = nn.ELU()
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



class CalibModel(nn.Module):
    """
    Model based on the nvidia architecture model for self driving. 
    I added temporal dependencies with lstm.
    """
    def __init__(self, img_size: list[int], consecutive_frames: int):
        super(CalibModel, self).__init__()

        # block_arg1 = BlockArgs(1, 64, 128, 3, 2)
        # block_arg2 = BlockArgs(1, 128, 128, 3, 1)

        #Nvidia basic as starting model.
        self.cnn = nn.Sequential(collections.OrderedDict([
            ("bn0", nn.BatchNorm2d(3)),
            
            ("conv1", nn.Conv2d(3, 24, 5, 2, bias=True)),
            ("bn1", nn.BatchNorm2d(24)),
            ("elu1", nn.ELU()),
            # nn.MaxPool2d(3, stride=1),

            ("conv2", nn.Conv2d(24, 36, 5, 2, bias=True)),
            ("bn2", nn.BatchNorm2d(36)),
            ("elu2", nn.ELU()),

            ("conv3", nn.Conv2d(36, 48, 5, 2, bias=True)),
            ("bn3" ,nn.BatchNorm2d(48)),
            ("elu3", nn.ELU()),

            ("conv4", nn.Conv2d(48, 64, 3, 1, bias=True)),
            ("bn4", nn.BatchNorm2d(64)),
            ("elu4", nn.ELU()),

            ("conv5", nn.Conv2d(64, 64, 3, 1, bias=True)),
            ("bn5", nn.BatchNorm2d(64)),
            ("elu5", nn.ELU()),

            ("avg_pool", nn.AvgPool2d(3, stride=1)),
            ("flatten", nn.Flatten(1))
        ]))   
        
        h, w = img_size
        out = self.extract_features(torch.zeros(1, 1, 3, h, w))
        self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))

        self.hidden_size = int(256) #temporary choose 
        self.lstm = nn.LSTM(self.cnn_shape_out, self.hidden_size, num_layers=3)

        # self.dropout = nn.Dropout(0.5)
        self.linear = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.hidden_size * consecutive_frames, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, consecutive_frames * 2)
        )

    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Process batch at each time step.
        """
        b, t, c, h, w = X.shape

        out = []
        for i in range(t):
            out.append(self.cnn(X[:, i]))

        #check if torch.stack cause problem or is slow 
        # https://discuss.pytorch.org/t/torch-stack-is-very-slow-and-cause-cuda-error/28554
        return torch.stack(out)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.extract_features(X)
        out, (hn, cn) = self.lstm(out)
        # out = self.dropout(out)
        out = out.permute(1, 0, 2)
        out = self.linear(out)
        return out