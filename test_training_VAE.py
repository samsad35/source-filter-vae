import unittest
import torch
from sf_vae import WSJ
from sf_vae import VAE
from sf_vae import WSJDataset, Train


class TestAudioTools(unittest.TestCase):
    def test_wsj(self):
        wsj = WSJ(directory_name=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s")
        wsj.create_h5(dir_save=r"H5", restart=True, section="train")

        wsj = WSJ(directory_name=r"D:\These\data\Audio\WSJ0\wsj0_si_dt_05")
        wsj.create_h5(dir_save=r"H5", restart=True, section="validation")

    def test_wsjdataset(self):
        wsj_train = WSJDataset(directory_name=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s", section="train",
                               h5_path=r"H5")
        print(wsj_train[0].shape)

    def test_wsjTrain(self):
        wsj_train = WSJDataset(directory_name=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s", section="train",
                               h5_path=r"H5")
        wsj_validation = WSJDataset(directory_name=r"D:\These\data\Audio\WSJ0\wsj0_si_dt_05", section="validation",
                                    h5_path=r"H5")

        vae = VAE()
        config_training = dict(batch_size=512, epochs=200, device="cuda", learning_rate=1e-3)
        train_vae = Train(vae, wsj_train, wsj_validation, config_training=config_training)
        train_vae()
