
# Learning and controlling the source-filter representation of speech with a variational autoencoder
[![Generic badge](https://img.shields.io/badge/<STATUS>-<in_progress>-<COLOR>.svg)](https://github.com/samsad35/source-filter-vae)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://tinyurl.com/iclr2022)
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://test.pypi.org/project/sf-vae/)
## Abstract

Understanding and controlling latent representations in deep generative models is a challenging yet important problem 
for analyzing, transforming and generating various types of data. In speech processing, inspiring from the anatomical 
mechanisms of phonation, the source-filter model considers that speech signals are produced from a few independent and 
physically meaningful continuous latent factors, among which the fundamental frequency and the formants are of primary 
importance. In this work, we show that the source-filter model of speech production naturally arises in the latent space
of a variational autoencoder (VAE) trained in an unsupervised fashion on a dataset of natural speech signals. Using only
a few seconds of labeled speech signals generated with an artificial speech synthesizer, we experimentally demonstrate
that the fundamental frequency and formant frequencies are encoded in orthogonal subspaces of the VAE latent space and
we develop a weakly-supervised method to accurately and independently control these speech factors of variation within 
the learned latent subspaces. Without requiring additional information such as text or human-labeled data, we propose a
deep generative model of speech spectrograms that is conditioned on the fundamental frequency and formant frequencies,
and which is applied to the transformation of speech signals.

- A link to see the [qualitative results](https://tinyurl.com/iclr2022).

## Setup 
- [x] Pypi:  
  - ```pip install -i https://test.pypi.org/simple/ sf-vae --no-deps```
- [x] Install the package locally (for use on your system):  
  - In source-filter-vae directoy: ```pip install -e .```
- [x] Virtual Environment: 
  - ```conda create -n sf_vae python=3.8```
  - ```conda activate sf_vae```
  - In source-filter-vae directoy: ```pip install -r requirements.txt```

## Usage
### LEARNING LATENT SUBSPACES ENCODING SOURCE-FILTER FACTORS OF VARIATION 
```python
import torch
from sf_vae import Learning
from sf_vae import VAE

vae = VAE()
checkpoint = torch.load(r"checkpoints\vae_trained")
vae.load_state_dict(checkpoint['model_state_dict'])
learn = Learning(config_factor=dict(factor="f1", path_trajectory="formant_1\\f2-1600", dim=3),
                 # f0: pitch (source), f1, f2, f3: formants (filter)
                model=vae,
                path_save=r"checkpoints\pca-regression")
learn()
```

### CONTROLLING THE FACTORS OF VARIATION FOR SPEECH TRANSFORMATION
```python
import torch
from sf_vae import Controlling
from sf_vae import VAE

vae = VAE()
checkpoint = torch.load(r"checkpoints\vae_trained")
vae.load_state_dict(checkpoint['model_state_dict'])
control = Controlling(path=r"checkpoints\pca-regression",
                    model=vae,
                    device="cuda")
control(path_wav=r"01aa0101.wav", 
        factor='f0', # f0: pitch (source), f1, f2, f3: formants (filter)
        y=(85, 300)) # The new values of the factor in Hz
```
* Phase reconstruction method:
  * [x] RTISI_LA
  * [ ] Griffin_lim
  * [ ] WaveGlow
* Whispering
```python
import torch
from sf_vae import Controlling
from sf_vae import VAE

vae = VAE()
checkpoint = torch.load(r"checkpoints\vae_trained")
vae.load_state_dict(checkpoint['model_state_dict'])
control = Controlling(path=r"checkpoints\pca-regression",
                    model=vae,
                    device="cuda")
z_ = control.whispering(path_wav=r"01aa0101.wav")
control.reconstruction(z_, save=True)
```

## GUI: graphic interface
```python
from sf_vae import Interface
from sf_vae import VAE
import torch

vae = VAE()
checkpoint = torch.load(r"checkpoints\vae_trained")
vae.load_state_dict(checkpoint['model_state_dict'])
inter = Interface(device="cuda", model=vae, path=r"checkpoints\pca-regression")
inter.master.mainloop()
```
![image](images/interface.jpeg)

## License
GNU Affero General Public License (version 3), see LICENSE.txt.