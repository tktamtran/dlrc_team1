

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network00(nn.Module):

    def __init__(self):
        """
        Class constructor which preinitializes NN layers with trainable
        parameters.
        """
        #super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forwards the input x through each of the NN layers and outputs the result.
        """
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # An efficient transition from spatial conv layers to flat 1D fully
        # connected layers is achieved by only changing the "view" on the
        # underlying data and memory structure.
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet()
print(net)