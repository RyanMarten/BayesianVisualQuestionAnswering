import os

import numpy as np
import torch
from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pickle

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        hidden = self.softplus(self.fc1(x))
        mu = self.softplus(self.fc21(hidden))
        logsigma = self.softplus(self.fc22(hidden))
        
        return mu, logsigma


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        out = self.sigmoid(self.fc2(hidden))
        
        return out
