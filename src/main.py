from torch import nn
import torch
from model import Endurance, ConvBlock
import utils as ut



# (291, 388)  (diviso 3)
glob_params = ut.GlobalParams(1, 1, (291, 388), 
                            (ut.conv_layer(3, 32, 5, 2, 2, 2),),
                          nn.AdaptiveAvgPool2d((1, 1)),
                          2, 256, 0.2,
                          0.3,
                          (128, 64, 16, 2))


ba1 = ut.BlockArgs(2, 32, 64, 3, 2, nn.ReLU())
ba2 = ut.BlockArgs(2, 64, 128, 3, 2, nn.ReLU())
ba3 = ut.BlockArgs(2, 128, 256, 3, 2, nn.ReLU())
ba4 = ut.BlockArgs(2, 256, 512, 3, 2, nn.ReLU())
ba5 = ut.BlockArgs(2, 512, 1024, 3, 2, nn.ReLU())

blocks_args = [ba1, ba2, ba3, ba4, ba5]

m = Endurance(blocks_args, glob_params)
