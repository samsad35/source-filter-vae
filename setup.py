from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name="sf_vae", version='0.3',
      description="Learning and controlling the source-filter representation of speech with a variational autoencoder",
      url="https://github.com/samsad35/source-filter-vae",
      project_url="https://sites.google.com/view/sturnus",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Samir Sadok",
      author_mail="samir.sadok@centralesupelec.fr",
      licence="MIT",
      packages=find_packages(where='sf_vae.sf_vae'),
      install_requires=["librosa>=0.8.0", "torch>=1.7.1", "sounddevice>=0.4.1", "scipy>=1.7.1", "numpy>=1.21.5",
                        "praat-parselmouth", "pwlf>=2.0.4", "torch_specinv>=0.1"],
      python_requires=">=3.6",
      zip_safe=False)



