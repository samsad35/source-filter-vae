import pickle
import numpy as np
import torch
from scipy.signal import get_window
from librosa.util import pad_center, tiny
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
        """

        :param path:
        :return:
        """
        signal, fs = self.tools.load(path, resample=16000)
        assert len(signal.shape) == 1, "Dimension of signal is incorrect"
        mel, magnitude, phase = self.tools.stft(signal)
        _, mu, var, z = self.model(torch.transpose(magnitude ** 2, 0, 1).to(self.device))
        return z.detach().cpu().numpy()

    def load_models(self, factor):
        """
            Load PCA + Regression parameters.
        :param factor: f0 -> (source), f1, f2, f3 -> (filter).
        :return: void.
        """
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

    def transform(self, path_wav: str = None, factor: str = None, y=None):
        r"""
            CONTROLLING THE FACTORS OF VARIATION FOR SPEECH TRANSFORMATION
        :param path_wav: The path to .wav file.
        :param factor: f0 -> (source), f1, f2, f3 -> (filter).
        :param y: The new values of the factor in Hz.
        :return: The new Z.
        """
        assert factor in ['f1', 'f2', 'f3', 'f0'], "Name of factor problem."
        self.load_models(factor)
        z = self.get_z(path_wav)
        if type(y) == tuple:
            y = np.linspace(y[0], y[1], z.shape[0])
        g = self.regression(y)
        if y.any():
            z_ = z - self.ipca(self.pca(z)) + self.ipca(np.array(g))
            return z_
        else:
            return z

    def reconstruction(self, z, save: bool = False, path_new_wav: str = "out.wav"):
        """
            Get the audio signal from latent space vae Z
        :param z: latent space.
        :param save: True if you would to save the output signal audio.
        :param path_new_wav: the path/name of the new output signal audio if save == True.
        :return: the output signal audio
        """
        spec = self.model.decode(torch.from_numpy(z).type(torch.FloatTensor).to(self.device))
        spec = torch.sqrt(spec)
        spec = torch.transpose(spec, 0, 1).detach().cpu().numpy()
        window = get_window(window=self.tools.stft_waveglow.window,
                            Nx=self.tools.stft_waveglow.win_length, fftbins=True)
        window = pad_center(window, self.tools.stft_waveglow.filter_length)
        window = torch.from_numpy(window).float()

        signal_recons = RTISI_LA(torch.from_numpy(spec),
                                 hop_length=self.tools.stft_waveglow.hop_length,
                                 win_length=self.tools.stft_waveglow.win_length,
                                 window=window).numpy()
        if save:
            write(path_new_wav, 16000, signal_recons)
        return signal_recons, spec

    def __call__(self, path_new_wav: str = "out.wav", *args, **kwargs):
        z_ = self.transform(**kwargs)
        signal_ = self.reconstruction(z_, save=True, path_new_wav=path_new_wav)
        return signal_

    def whispering(self, path_wav: str = None):
        """
            Whispering
        :param path_wav: the path to .wav file.
        :return: The new Z.
        """
        self.load_models('f0')
        z = self.get_z(path_wav)
        z_ = z - self.ipca(self.pca(z))
        return z_
