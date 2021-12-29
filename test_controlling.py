import unittest
import torch
from sf_vae import Controlling
from sf_vae import VAE


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
        control(path_wav=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa0101.wav",
                y=(85, 300), factor='f0', path_new_wav='new.wav')


if __name__ == '__main__':
    unittest.main()
