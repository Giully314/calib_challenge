import torch
from torch import nn

from utils import (
    conv1x1, 
    conv,
    BlockArgs,
    GlobalParams,
)




class ConvBlock(nn.Module):
    def __init__(self, block_args: BlockArgs):
        super(ConvBlock, self).__init__()
        self.block_args = block_args

        self.conv_block = nn.Sequential(
            conv(self.block_args.in_channel, self.block_args.out_channel, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channel),
            self.block_args.activation_fun,
            conv(self.block_args.out_channel, self.block_args.out_channel),
            nn.BatchNorm2d(self.block_args.out_channel),
        )
        
        self.activation_fun = self.block_args.activation_fun
        self.proj = nn.Sequential(
            conv1x1(self.block_args.in_channel, self.block_args.out_channel, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channel))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv_block(out)
        out += identity
        out = self.activation_fun(out)
        
        return out

    

class Endurance(nn.Module):
    def __init__(self, blocks_args: list[BlockArgs]):
        super(Endurance, self).__init__()
        self.block_args = block_args
        

        