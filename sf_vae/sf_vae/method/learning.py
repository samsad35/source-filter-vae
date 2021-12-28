import torch
from matplotlib import pyplot as plt
import librosa
import pwlf
import numpy as np
from ..utils import AudioTools
import pickle
import warnings
warnings.filterwarnings(action='ignore')


class Learning:
    mu: np.ndarray
    mean: np.ndarray
    s: np.ndarray
    U: np.ndarray
    eigenvalues: np.ndarray
    label: list
    __ratio: list

    def __init__(self, config_factor: dict = None, path_save: str = None,
                 model=None, device='cuda'):
        # Information about factor:
        self.config_factor = config_factor
        self.factor = self.config_factor['factor']
        factors = ["f1", "f2", "f3", "f0"]
        assert self.factor in factors, "The name of the factor should be f1 / f2 / f3 / f0."
        self.dim = self.config_factor['dim']
        self.path_trajectory = self.config_factor['path_trajectory']
        assert bool(self.path_trajectory), "path to trajectory should be defined in the config_factor."
        self.path_save = path_save
        # Model :
        self.model = model
        # Device:
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        # Audio tools:
        self.tools = AudioTools()

    def generator(self):
        files = librosa.util.find_files(self.path_trajectory, ext='wav')
        for file in files:
            label = float(file.split('.')[0].split("\\")[-1].split('-')[-1])
            yield file, label

    def encoder(self, path: str, limitation: tuple = None):
        """

        :param path:
        :param limitation:
        :return:
        """
        signal, fs = self.tools.load(path)
        assert len(signal.shape) == 1, "Dimension of signal is incorrect"
        if limitation is not None:
            signal = signal[limitation[0]:limitation[1]]
        mel, magnitude, phase = self.tools.stft(signal)
        _, mu, var, z = self.model(torch.transpose(magnitude ** 2, 0, 1).to(self.device))
        return mu.detach().cpu(), var.detach().cpu(), z.detach().cpu()

    def get_trajectory(self):
        """

        :return:
        """
        self.mu, self.var, self.label = [], [], []
        for file, label in self.generator():
            mu, var, z = self.encoder(file, limitation=(3000, 6000))
            self.mu.append(torch.mean(mu, 0).numpy())
            self.var.append(torch.mean(var, 0).numpy())
            self.label.append(label)
        self.mu = np.array(self.mu)
        self.var = np.array(self.var)

    def get_s(self):
        """

        :return:
        """
        self.get_trajectory()
        num_points = len(self.label)
        self.s = np.zeros((16, 16))
        for i in range(num_points):
            temp = np.array([self.mu[i]]).T @ np.array([self.mu[i]]) + np.diag(np.exp(self.var[i]))
            self.s = self.s + temp
        self.mean = np.mean(self.mu, axis=0)
        self.s = self.s/num_points - np.array([self.mean]).T @ np.array([self.mean])

    def get_u(self):
        """

        :return:
        """
        self.get_s()
        w, V = np.linalg.eigh(self.s)
        eigenvalues = w[-1::-1]  # an decreasing ordering of the eigenvalues
        eigenvectors = V[:, -1::-1]
        tot = np.sum(eigenvalues)
        self.U = eigenvectors[:, :self.dim]
        self.eigenvalues = eigenvalues[:self.dim]
        self.__ratio = [round((value * 100/tot)/100, 8) for value in self.eigenvalues]
        print(f' \t [factor = {self.factor}] [dim = {self.dim}] [variance retained = {np.sum(self.__ratio) * 100}]')
        with open(f"{self.path_save}\\pca_{self.factor}", 'wb') as f:
            pickle.dump(dict(u=self.U, eigenvalues=self.eigenvalues, ratio=self.__ratio, mean=self.mean), f)

    def pca(self, data: np.ndarray):
        """

        :param data:
        :return:
        """
        return (data - self.mean) @ self.U

    def ipca(self, p: np.ndarray):
        """

        :return:
        """
        return (p @ self.U.T) + self.mean

    def regression(self, dim: int = 0, num_segments: int = 3):
        """

        :return:
        """
        x = self.pca(self.mu)
        my_pwlf = pwlf.PiecewiseLinFit(self.label, x[:, dim])
        my_pwlf.fitfast(num_segments)
        with open(f"{self.path_save}\\pwl_{self.factor}_axe{dim}", 'wb') as f:
            pickle.dump(my_pwlf, f)

    def __call__(self, *args, **kwargs):
        self.get_u()
        for i in range(self.dim):
            self.regression(dim=i, num_segments=20)
