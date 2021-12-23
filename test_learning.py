import unittest
import torch
from src import Learning
from src import VAE


class TestAudioTools(unittest.TestCase):
    def test_get_s(self):
        vae = VAE()
        learn = Learning(config_factor=dict(factor="f1",
                                            path_trajectory=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\formant_1\f2-1600",
                                            dim=3), model=vae)
        # learn.get_trajectory()
        learn.get_s()
        self.assertEqual(learn.s.shape, (16, 16))

    def test_get_u(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        learn = Learning(config_factor=dict(factor="f1",
                                            path_trajectory=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\formant_1\f2-1600",
                                            dim=3), model=vae, path_save=r"checkpoints\pca-regression")
        # learn.get_u()

    def test_learning(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        learn = Learning(config_factor=dict(factor="f1",
                                            path_trajectory=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\formant_1\f2-1600",
                                            dim=3), model=vae, path_save=r"checkpoints\pca-regression")
        learn()


if __name__ == '__main__':
    unittest.main()
