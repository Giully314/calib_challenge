import torch
from torch import nn
import functools 
import operator
import collections


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target) + 1e-8)


CalibParams = collections.namedtuple("CalibParams", [
    "img_size", "consecutive_frames", "lstm_hidden_size", "lstm_num_layers", "lstm_dropout", 
    "linear_layers", "linear_dropout"
])
CalibParams.__new__.__defaults__ = (None,) * len(CalibParams._fields)



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
    def __init__(self, calib_params: CalibParams):
        super(CalibModel, self).__init__()

        #Nvidia basic as starting model.
        self.cnn = nn.Sequential(collections.OrderedDict([
            ("bn0", nn.BatchNorm2d(3)),
            
            ("conv1", nn.Conv2d(3, 24, 5, 2, bias=False)),
            ("bn1", nn.BatchNorm2d(24)),
            ("elu1", nn.ELU()),
            # ("max1", nn.MaxPool2d(3, stride=2)),

            ("conv2", nn.Conv2d(24, 36, 5, 2, bias=False)),
            ("bn2", nn.BatchNorm2d(36)),
            ("elu2", nn.ELU()),

            ("conv3", nn.Conv2d(36, 48, 5, 2, bias=False)),
            ("bn3" ,nn.BatchNorm2d(48)),
            ("elu3", nn.ELU()),

            ("conv4", nn.Conv2d(48, 64, 3, 1, bias=False)),
            ("bn4", nn.BatchNorm2d(64)),
            ("elu4", nn.ELU()),

            ("conv5", nn.Conv2d(64, 64, 3, 1, bias=False)),
            ("bn5", nn.BatchNorm2d(64)),
            ("elu5", nn.ELU()),

            ("avg_pool", nn.AvgPool2d(5, stride=2)),
            ("flatten", nn.Flatten(1))
        ]))   
        
        h, w = calib_params.img_size
        out = self.extract_features(torch.zeros(1, 1, 3, h, w))
        # print(f"OUT SHAPE {out.shape}")
        self.cnn_shape_out = functools.reduce(operator.mul, list(out.shape))

        #TODO BEFORE USE LSTM, LEARN WHAT IT IS, IDIOT.
        self.hidden_size = int(calib_params.lstm_hidden_size) #temporary choose 
        self.lstm = nn.LSTM(self.cnn_shape_out, self.hidden_size, num_layers=calib_params.lstm_num_layers,
                            dropout=calib_params.lstm_dropout)

        consecutive_frames = calib_params.consecutive_frames

        #self.dropout = nn.Dropout(calib_params.linear_dropout)

        self.linear = nn.ModuleList([])
        self.linear.append(nn.Flatten(1)) #??? it's the right way to process the output of the lstm?
        
        linear_layers = [self.hidden_size * consecutive_frames]  + list(calib_params.linear_layers) + [consecutive_frames * 2]
        for i in range(len(linear_layers) - 2):
            self.linear.append(nn.Linear(linear_layers[i], linear_layers[i + 1])) #regulate this layer
            self.linear.append(nn.ELU())
            self.linear.append(nn.Dropout(calib_params.linear_dropout))
            
        self.linear.append(nn.Linear(linear_layers[-2], linear_layers[-1]))


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

    def process_information(self, X: torch.Tensor) -> torch.Tensor:
        out = X
        for layer in self.linear:
            out = layer(out)
        return out


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.extract_features(X)
        
        out, (hn, cn) = self.lstm(out)
       
        # out = self.dropout(out)
        out = out.permute(1, 0, 2)
    
        out = self.process_information(out)

        return out