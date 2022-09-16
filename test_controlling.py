import unittest
import torch
from sf_vae import Controlling
from sf_vae import VAE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
sns.set()


class TestAudioTools(unittest.TestCase):
    def test_load(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        control = Controlling(path=r"checkpoints\pca-regression", model=vae, device="cuda")
        control.load_models(factor='f0')

    def test_transform(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        control = Controlling(path=r"checkpoints\pca-regression", model=vae, device="cuda")


        # z = control.test_(path_wav=r"D:\These\Sites\site-sfvae\demos\sf_vae\controlling\f0-gaussian.wav")
        # control.reconstruction(z, save=True, method_reconstruction="WAVEGLOW")
        # control(path_wav=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa0101.wav",
        #         y=(85, 300), factor='f0', path_new_wav='new.wav')


if __name__ == '__main__':
    unittest.main()
