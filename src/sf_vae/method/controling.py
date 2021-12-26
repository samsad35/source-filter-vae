import pickle
import numpy as np
import torch
from torch_specinv.methods import RTISI_LA
from scipy.io.wavfile import write
from ..utils import AudioTools


class Controlling:
    mean: np.ndarray
    U: np.ndarray
    eigenvalues: np.ndarray
    __ratio: list
    pwl: list

    def __init__(self, path: str = None, model=None, device: str = None):
        self.path = path
        # Device:
        self.device = device
        # Model: (VAE)
        self.model = model
        self.model = self.model.to(self.device)
        # Tools :
        self.tools = AudioTools()

    def get_z(self, path: str):
        signal, fs = self.tools.load(path, resample=16000)
        assert len(signal.shape) == 1, "Dimension of signal is incorrect"
        mel, magnitude, phase = self.tools.stft(signal)
        _, mu, var, z = self.model(torch.transpose(magnitude ** 2, 0, 1).to(self.device))
        return z.detach().cpu().numpy()

    def load_models(self, factor):
        with open(f"{self.path}\\pca_{factor}", 'rb') as f:
            pca = pickle.load(f)
            self.U, self.mean, self.eigenvalues, self.__ratio = pca['u'], pca['mean'], pca['eigenvalues'], pca['ratio']
        self.pwl = []
        for i in range(len(self.eigenvalues)):
            with open(f"{self.path}\\pwl_{factor}_axe{i}", 'rb') as f:
                self.pwl.append(pickle.load(f))
        print(f"Models pca + Regression loaded successfully [{factor}, ratio: {np.sum(self.__ratio) * 100}] ")

    def pca(self, data: np.ndarray) -> np.ndarray:
        """
            The PCA transform.
        :param data:
        :return:
        """
        return (data - self.mean) @ self.U

    def ipca(self, p: np.ndarray) -> np.ndarray:
        """
            The inverse PCA transform.
        :return:
        """
        return (p @ self.U.T) + self.mean

    def regression(self, y: np.ndarray):
        p = []
        for t in y:
            point = []
            for m in self.pwl:
                point.append(m.predict([[int(t)]])[0])
            p.append(point)
        return p

    def transform(self, path_wav: str = None, factor: str = None, y: tuple = None):
        assert factor in ['f1', 'f2', 'f3', 'f0'], "Name of factor problem."
        self.load_models(factor)
        z = self.get_z(path_wav)
        y = np.linspace(y[0], y[1], z.shape[0])
        g = self.regression(y)
        z_ = z - self.ipca(self.pca(z)) + self.ipca(np.array(g))
        return z_

    def reconstruction(self, z, save: bool = False):
        spec = self.model.decode(torch.from_numpy(z).type(torch.FloatTensor).to(self.device))
        spec = torch.sqrt(spec)
        spec = torch.transpose(spec, 0, 1).detach().cpu().numpy()
        signal_recons = RTISI_LA(torch.from_numpy(spec), maxiter=50).numpy()
        if save:
            write("out.wav", 16000, signal_recons)
        return signal_recons

    def __call__(self, *args, **kwargs):
        z_ = self.transform(**kwargs)
        signal_ = self.reconstruction(z_, save=True)
        return signal_
