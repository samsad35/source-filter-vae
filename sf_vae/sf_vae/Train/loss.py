from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class Loss:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='sum')

    @staticmethod
    def mse_(x, y):
        """
            Mean Square error
        :param x:
        :param y:
        :return:
        """
        ret = torch.sum((x - y) ** 2)
        return ret

    @staticmethod
    def kl_divergence(mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss

    @staticmethod
    def is_divergence(x, recon_x):
        loss = torch.sum(x / recon_x - torch.log(x / recon_x) - 1)
        return loss

    def __call__(self, x, x_, mu, logvar, beta=1, batch_size=32):
        loss = self.is_divergence(x, x_) + beta*self.kl_divergence(mu, logvar)
        return loss
