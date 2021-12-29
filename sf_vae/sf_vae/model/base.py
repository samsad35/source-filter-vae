import torch.nn as nn


class BASE(nn.Module):
    def __init__(self):
        super(BASE, self).__init__()

    def encode(self, x) -> tuple:
        pass

    def decode(self, z) -> tuple:
        pass

    def forward(self, x) -> tuple:
        pass

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1) -> dict:
        pass
