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

# This code is heavily based on the Pyro tutorial code for Variational Auto-Encoders: https://pyro.ai/examples/vae.html

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_dim, z_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(5184, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = self.pool(self.softplus(self.conv1(x)))
        x = self.pool(self.softplus(self.conv2(x)))
        x = x.flatten(1)
        x = self.softplus(self.fc1(x))
        
        mu = self.fc21(x)
        sigma = torch.exp(self.fc22(x))
        
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_channels, output_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.fc1 = nn.Linear(z_dim, hidden_channels[1] * (75 // 4)**2)
        self.convt1 = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=3, stride=2)
        self.convt2 = nn.ConvTranspose2d(hidden_channels[0], 3, kernel_size=3, stride=2)
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        x = self.softplus(self.fc1(z))
        x = x.view(x.shape[0], self.hidden_channels[1], 75 // 4, 75 // 4)
        x = self.softplus(self.convt1(x))
        x = self.sigmoid(self.convt2(x))
        return x

class VAE(nn.Module):
    
    def __init__(self, z_dim=64, hidden_dim=2048, hc=(8,16), use_cuda=False):
        super().__init__()
        self.encoder = Encoder(input_channels=3, hidden_channels=hc, hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, hidden_channels=hc, output_channels=3)
        
        if use_cuda:
            self.cuda()
            
        self.use_cuda = use_cuda
        self.z_dim = z_dim
    
    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder(z)
            x_flat = x.flatten(1)
            loc_img_flat = loc_img.flatten(1)
            obs = pyro.sample("obs", dist.Bernoulli(loc_img_flat).to_event(1), obs=x_flat.reshape(-1, 16875))
            
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        
    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img