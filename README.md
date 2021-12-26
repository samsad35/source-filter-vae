# Learning and controlling the source-filter representation of speech with a variational autoencoder

---

## Abstract

Understanding and controlling latent representations in deep generative models is a challenging yet important problem for analyzing, transforming and generating various types of data. In speech processing, inspiring from the anatomical mechanisms of phonation, the source-filter model considers that speech signals are produced from a few independent and physically meaningful continuous latent factors, among which the fundamental frequency and the formants are of primary importance. In this work, we show that the source-filter model of speech production naturally arises in the latent space of a variational autoencoder (VAE) trained in an unsupervised fashion on a dataset of natural speech signals. Using only a few seconds of labeled speech signals generated with an artificial speech synthesizer, we experimentally demonstrate that the fundamental frequency and formant frequencies are encoded in orthogonal subspaces of the VAE latent space and we develop a weakly-supervised method to accurately and independently control these speech factors of variation within the learned latent subspaces. Without requiring additional information such as text or human-labeled data, we propose a deep generative model of speech spectrograms that is conditioned on the fundamental frequency and formant frequencies, and which is applied to the transformation of speech signals.

A link to see the [qualitative results](https://tinyurl.com/iclr2022)

---
## Installation

> requirements.txt
>
> pip install sf_vae

---

## User utilisation
### Learning 
    from src import Learning
    from src import VAE
    
    vae = VAE()
    checkpoint = torch.load(r"checkpoints\vae_trained")
    vae.load_state_dict(checkpoint['model_state_dict'])
    learn = Learning(config_factor=dict(factor="f1", path_trajectory="\formant_1\f2-1600", dim=3),
                    model=vae,
                    path_save=r"checkpoints\pca-regression")
    learn()
    
    
    
### Controlling 
    from src import Controlling
    from src import VAE
    
    vae = VAE()
    checkpoint = torch.load(r"checkpoints\vae_trained")
    vae.load_state_dict(checkpoint['model_state_dict'])
    control = Controlling(path=r"checkpoints\pca-regression",
                            model=vae,
                            device="cuda")
    control(path_wav=r"01aa0101.wav", y=(85, 300), factor='f0')

---