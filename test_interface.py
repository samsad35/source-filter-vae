from sf_vae import Interface
from sf_vae import VAE
import torch

vae = VAE()
checkpoint = torch.load(r"checkpoints\vae_trained")
vae.load_state_dict(checkpoint['model_state_dict'])
inter = Interface(device="cuda", model=vae, path=r"checkpoints\pca-regression")
inter.master.mainloop()

