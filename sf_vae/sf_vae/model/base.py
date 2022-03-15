import torch.nn as nn
from abc import ABC, abstractmethod
import torch


class BASE(ABC, nn.Module):
    def __init__(self):
        super(BASE, self).__init__()

    @abstractmethod
    def encode(self, x) -> tuple:
        pass

    @abstractmethod
    def decode(self, z) -> tuple:
        pass

    @abstractmethod
    def forward(self, x) -> tuple:
        pass

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1) -> dict:
        pass


