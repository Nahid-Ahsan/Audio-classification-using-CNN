import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init 



class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # initialize first convolution block with relu and batch normalization
        self.conv1 = nn.Conv2d(2, 8, kernel_size = (5,5), stride = (2,2), padding = (2,2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a = 0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Initialize second convolution block
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (3,3), stride = (2,2), padding = (1,1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a = 0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Initialize third convolution block
        self.conv3 = nn.Conv2d(16, 32, kernel_size = (3,3), stride = (2,2), padding = (1,1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a = 0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Initialize fouth convolution block
        self.conv4 = nn.Conv2d(32, 64,  kernel_size = (3,3), stride = (2,2), padding = (1,1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a = 0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # linear classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size = 1)
        self.lin = nn.Linear(in_features = 64, out_features = 10)


        # wrap the convolutional block 
        self.conv = nn.Sequential(*conv_layers)


    def forward(self, x):
        x = self.conv(x)  # run convolutional block
        x = self.ap(x)  # adaptive pooling 
        x = x.view(x.shape[0], -1) # flatten for input to linear layer 
        x = self.lin(x) # linear layer

        return x 

