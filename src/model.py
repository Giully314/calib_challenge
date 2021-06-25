import torch
from torch import nn

from utils import (
    conv1x1, 
    conv,
    BlockArgs,
    GlobalParams,
    round_filters,
    round_repeats,
    calculate_output_image_size,
    get_width_and_height,
)




class ConvBlock(nn.Module):
    def __init__(self, block_args: BlockArgs):
        super(ConvBlock, self).__init__()
        self.block_args = block_args

        self.conv_block = nn.Sequential(
            conv(self.block_args.in_channels, self.block_args.out_channels, self.block_args.kernel_size, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channels),
            self.block_args.activation_fun,
            conv(self.block_args.out_channels, self.block_args.out_channels, self.block_args.kernel_size,),
            nn.BatchNorm2d(self.block_args.out_channels),
        )
        
        self.activation_fun = self.block_args.activation_fun
        self.proj = nn.Sequential(
            conv1x1(self.block_args.in_channels, self.block_args.out_channels, self.block_args.stride),
            nn.BatchNorm2d(self.block_args.out_channels))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv_block(out)
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

        self.first_cnn_layer = nn.Sequential(
            nn.Conv2d(self.global_params.in_channels, self.global_params.out_channels, self.global_params.kernel_size, 
            self.global_params.stride),
            nn.BatchNorm2d(self.global_params.out_channels),
            nn.ReLU(),  
        )

        self.conv_blocks = nn.ModuleList([])

        img_size = self.global_params.image_size
    
        for block_args in self.blocks_args:
            block_args = block_args._replace(
                in_channels=round_filters(block_args.in_channels, self.global_params),
                out_channels=round_filters(block_args.out_channels, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )
            self.conv_blocks.append(ConvBlock(block_args))

            img_size = calculate_output_image_size(img_size, block_args.stride)


            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(in_channels=block_args.out_channels, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self.conv_blocks.append(ConvBlock(block_args))

        cnn_shape_out = img_size[0] * img_size[1] * self.blocks_args[-1].out_channels

        self.pooling_layer = self.global_params.final_pooling_layer
        self.flatten = nn.Flatten(1)

        self.hidden_dim = self.global_params.hidden_dim
        self.lstm_num_layers = self.global_params.num_recurrent_layers
        self.lstm = nn.LSTM(cnn_shape_out, self.hidden_dim, num_layers=self.lstm_num_layers, dropout=self.global_params.rec_dropout)
        self.hidden_state = torch.zeros(self.lstm_num_layers, 1, self.hidden_dim, device=dev)
        self.cell_state = torch.zeros(self.lstm_num_layers, 1, self.hidden_dim, device=dev)


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
            out = self.first_cnn_layer(x[:, i])

            for conv_block in self.conv_blocks:
                out = self.conv_blocks(out)

            processed_frames.append(out)

        return torch.stack(processed_frames)

    def forward(self, x):
        return extract_features(x)

    
    def zero_hidden(self):
        self.hidden_state, self.cell_state = (torch.zeros(self.lstm_num_layers, 1, self.hidden_dim, device=dev),
                                                torch.zeros(self.lstm_num_layers, 1, self.hidden_dim, device=dev))