import unittest
from sf_vae import VAE
import torch


class TestAudioTools(unittest.TestCase):
    def test_model(self):
        vae = VAE()
        x_recon, mu, logvar, z = vae(torch.randn(513))
        self.assertEqual(x_recon.shape[-1], 513)
        self.assertEqual(z.shape[-1], 16)

    def test_load_model(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        loss = checkpoint['loss']
        print(f'\t  * Model loaded successfully  [loss function = {loss}]')
        vae.load_state_dict(checkpoint['model_state_dict'])
        self.assertIsNotNone(loss)


if __name__ == '__main__':
    unittest.main()
