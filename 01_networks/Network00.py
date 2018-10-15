

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class Network00(nn.Module):

    def __init__(self):

        super(Network00, self).__init__()
        self.fc1 = nn.Linear(15, 30)
        self.fc2 = nn.Linear(30, 60)
        self.fc3 = nn.Linear(60, 120)
        self.fc4 = nn.Linear(15, 8)
        self.fc5 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x) # 15 to 30
        x = self.fc2(x) # 30 to 60
        x = self.fc3(x) # 60 to 120
        x = F.max_pool1d(F.relu(x).unsqueeze(dim=0), 2) # 120 to 60
        x = F.max_pool1d(F.relu(x), 2) # 60 to 30
        x = F.avg_pool1d(F.relu(x), 2) # 30 to 15
        x = self.fc4(x) # 15 to 8
        x = F.max_pool1d(F.relu(x), 2) # 8 to 4
        x = self.fc5(x) # 4 to 1
        x = x.squeeze()
        x = x.squeeze()

        return x


class Network01(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(Network01, self).__init__()
        self.fc1 = nn.Linear(in_dim, 30)
        self.fc2 = nn.Linear(30, 60)
        self.fc3 = nn.Linear(60, 120)
        self.fc4 = nn.Linear(15, 8)
        self.fc5 = nn.Linear(4, out_dim)

    def forward(self, x):

        x = self.fc1(x) # in_dim to 30
        x = self.fc2(x) # 30 to 60
        x = self.fc3(x) # 60 to 120
        x = F.max_pool1d(F.relu(x).unsqueeze(dim=0), 2) # 120 to 60
        x = F.max_pool1d(F.relu(x), 2) # 60 to 30
        x = F.avg_pool1d(F.relu(x), 2) # 30 to 15
        x = self.fc4(x) # 15 to 8
        x = F.max_pool1d(F.relu(x), 2) # 8 to 4
        x = self.fc5(x) # 4 to out_dim
        x = F.relu(x)
        x = x.squeeze()
        x = x.squeeze()

        return x


class Network02(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(Network01, self).__init__()
        self.fc1 = nn.Linear(in_dim, 30)
        self.fc2 = nn.Linear(30, 60)
        self.fc3 = nn.Linear(60, 120)
        self.fc3a = nn.Linear(120,15)
        self.fc4 = nn.Linear(15, 8)
        self.fc5 = nn.Linear(4, out_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x)) # in_dim to 30
        x = F.relu(self.fc2(x)) # 30 to 60
        x = F.relu(self.fc3(x)) # 60 to 120
        x = F.relu(self.fc3a(x))
        x = F.relu(self.fc4(x)) # 15 to 8
        x = F.relu(self.fc5(x)) # 4 to out_dim
        #x += torch.tensor([5]*out_dim)
        x = x.squeeze()
        x = x.squeeze()

        return x




class LogProb_Loss(torch.nn.Module):

    def __init__(self):
        super(LogProb_Loss, self).__init__()

    def forward(self, mu, std, wcp):

        reg_weight_mu = 5e15
        reg_weight_std = 10e15

        std += 1e-6
        prob = Normal(mu, std).log_prob(wcp)
        loss = torch.mean(prob) + reg_weight_mu*torch.sum(mu**2) + reg_weight_std*torch.sum(std**2)
        loss *= -1
        return loss