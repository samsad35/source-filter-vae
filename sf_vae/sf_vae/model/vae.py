import torch
import torch.nn as nn


class VAE(nn.Module):
    """
        Class : Variational Auto Encoder (VAE) with only linear Layer ---> (Frame of spectrograme)
    """

    def __init__(self):
        super(VAE, self).__init__()

        # MLP :   x --> z : q(z|x)
        self.fc1 = nn.Linear(513, 256)
        self.fc2 = nn.Linear(256, 64)
        # -----
        self.fc31 = nn.Linear(64, 16)  # for the mean
        self.fc32 = nn.Linear(64, 16)  # for the log var
        # -----
        # MLP :   z --> x : p(x|z)
        self.fc4 = nn.Linear(16, 64)
        self.fc5 = nn.Linear(64, 256)
        self.fc6 = nn.Linear(256, 513)

        # Dropout :
        self.dropout = nn.Dropout(p=0.2)

        # Activation Function :
        self.activation = torch.nn.Tanh()

    def encode(self, x):
        """

        :param x:
        :return:
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc31(x), self.fc32(x)

    @staticmethod
    def reparameterize(mu, logvar):
        """

        :param mu:
        :param logvar:
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """

        :param z:
        :return:
        """
        z = self.activation(self.fc4(z))
        z = self.activation(self.fc5(z))
        z = self.fc6(z)
        return torch.exp(z)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1):
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        recon = torch.sum(x / recon_x - torch.log(x / recon_x) - 1)
        KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
        return {'loss_recon': recon, 'kld': KLD, 'loss_total': recon + beta * KLD}
