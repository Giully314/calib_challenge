import torch
from torch import nn
from torch.nn import functional as F
import functools 
import operator
import collections

from torch.nn.modules.linear import Linear 

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
            "activation_fun", 
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


class Conv3dBlock(nn.Module):
    def __init__(self, block_args: BlockArgs):
        super(ConvBlock, self).__init__()
        self.block_args = block_args

        self.conv = nn.Sequential(
            conv2d(self.block_args.in_channels, self.block_args.out_channels, self.block_args.kernel_size, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channels),
            self.block_args.activation_fun,
            conv2d(self.block_args.out_channels, self.block_args.out_channels, self.block_args.kernel_size,),
            nn.BatchNorm2d(self.block_args.out_channels),
        )
        
        self.activation_fun = self.block_args.activation_fun
        self.proj = nn.Sequential(
            conv1x1(self.block_args.in_channels, self.block_args.out_channels, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channels))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv(x)
        out += identity
        out = self.activation_fun(out)
        
        return out




class ConvBlock(nn.Module):
    def __init__(self, block_args: BlockArgs):
        super(ConvBlock, self).__init__()
        self.block_args = block_args

        self.conv = nn.Sequential(
            conv2d(self.block_args.in_channels, self.block_args.out_channels, self.block_args.kernel_size, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channels),
            self.block_args.activation_fun,
            conv2d(self.block_args.out_channels, self.block_args.out_channels, self.block_args.kernel_size,),
            nn.BatchNorm2d(self.block_args.out_channels),
        )
        
        self.activation_fun = self.block_args.activation_fun
        self.proj = nn.Sequential(
            conv1x1(self.block_args.in_channels, self.block_args.out_channels, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channels))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv(x)
        out += identity
        out = self.activation_fun(out)
        
        return out

    

class Endurance(nn.Module):
    def __init__(self, blocks_args: list[BlockArgs], global_params: GlobalParams):
        super(Endurance, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        
        self.blocks_args = blocks_args
        self.global_params = global_params

        self.conv_first_layers = nn.ModuleList([])

        for conv_layer in self.global_params.conv_layers:
            for layer in conv_layer:
                self.conv_first_layers.append(layer)
    
        self.conv_blocks = nn.ModuleList([])
        for i, block_args in enumerate(self.blocks_args):
            block_args = block_args._replace(
                in_channels=round_filters(block_args.in_channels, self.global_params),
                out_channels=round_filters(block_args.out_channels, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )
            self.conv_blocks.append(ConvBlock(block_args))

            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(in_channels=block_args.out_channels, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self.conv_blocks.append(ConvBlock(block_args))
        

        self.pooling_layer = self.global_params.final_pooling_layer
        self.flatten = nn.Flatten(1)
        h, w = self.global_params.image_size
        out = self.extract_features(torch.zeros(1, 1, 3, h, w))
        cnn_shape_out = functools.reduce(operator.mul, list(out.shape))

        self.hidden_dim = self.global_params.hidden_dim
        self.lstm_num_layers = self.global_params.num_recurrent_layers
        self.lstm = nn.LSTM(cnn_shape_out, self.hidden_dim, num_layers=self.lstm_num_layers, dropout=self.global_params.rec_dropout)
        self.hidden_state = torch.zeros(self.lstm_num_layers, 1, self.hidden_dim)
        self.cell_state = torch.zeros(self.lstm_num_layers, 1, self.hidden_dim)

        self.dropout = nn.Dropout(self.global_params.dropout)
        

        self.fc_layers = nn.ModuleList([])
        self.fc_layers.append(nn.Linear(self.hidden_dim, self.global_params.fc_layers[0]))
        for i, fc_layer in enumerate(self.global_params.fc_layers[1:-1]):
            self.fc_layers.append(nn.Linear(self.global_params.fc_layers[i-1], fc_layer))
            self.fc_layers.append(nn.ReLU())

        self.fc_layers.append(nn.Linear(self.global_params.fc_layers[-2], self.global_params.fc_layers[-1]))



    def extract_features(self, x):
        #Apply conv
        bs, ts, c, h, w = x.shape
        
        processed_frames = []
        for i in range(ts):
            out = x[:, i]
            for conv_layer in self.conv_first_layers:
                out = conv_layer(out)

            for conv_block in self.conv_blocks:
                out = conv_block(out)

            out = self.pooling_layer(out)
            out = self.flatten(out)

            processed_frames.append(out)

        return torch.stack(processed_frames)


    def forward(self, x):
        out = self.extract_features(x)
        out, _ = self.lstm(out)
        out = out.permute(1, 0, 2)
        
        out = out[:, -1]
        out = self.dropout(out)

        for fc_layer in self.fc_layers:
            out = fc_layer(out)

        return out

    #bad, DON'T DO THIS 
    def to(self, dev):
        super().to(dev)
        self.hidden_state = self.hidden_state.to(dev)
        self.cell_state = self.cell_state.to(dev)
    
    #todo: batch dimension
    def zero_hidden(self):
        self.hidden_state = torch.zeros(self.lstm_num_layers, 1, self.hidden_dim, device=self.hidden_state.device)
        self.cell_state = torch.zeros(self.lstm_num_layers, 1, self.hidden_dim, device=self.cell_state.device)



class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            
            nn.Conv2d(3, 24, 5, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            nn.Conv2d(24, 36, 5, 2),
            nn.BatchNorm2d(36),
            nn.ReLU(),

            nn.Conv2d(36, 48, 5, 2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(2),

            nn.Flatten(1)
        )
        out = self.cnn(torch.zeros(1, 3, 106, 350))
        self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))
        
        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Sequential(
            nn.Linear(self.cnn_shape_out, 100),
            nn.PReLU(), 
            nn.Linear(100, 50),
            nn.PReLU(),
            nn.Linear(50, 10),
            nn.PReLU(), 
            nn.Linear(10, 2),
        )

    def forward(self, X):
        out = self.cnn(X)
        out = self.dropout(out)
        out = self.linear(out)
        return out


def load_nvidia_model(model: nn.Module=None, path: str="model_state_dict.pt"):
    if model is None:
        model = NvidiaModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def get_nvidia_model_sgd(dev: torch.device = torch.device("cpu")):
    model = NvidiaModel()
    model.to(dev)
    #parameters found by random search using ray tune.
    opt = torch.optim.SGD(model.parameters(), lr=.023508264995070294, momentum=0.4785200241865478)
    return model, opt

def get_nvidia_model_sgd_sched(dev: torch.device = torch.device("cpu")):
    model, opt = get_nvidia_model_sgd(dev)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [10, 20], 0.1) #default found by empirical exp. (based on 30 epochs fit)
    return model, opt, scheduler




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