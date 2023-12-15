import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import data.GenreateTopologies as Top
import paths
from model.ModelParameters import ModelParameters
from dataclasses import dataclass
import model.train_util as train_util


class VFEncoder(nn.Module):  # Create AE class inheriting from pytorch nn Module class
    def __init__(self):
        super(VFEncoder, self).__init__()
        self.latent_dim = 1
    def forward(self, x):
        return x

