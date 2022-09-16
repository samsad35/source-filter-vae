import unittest
import matplotlib.pyplot as plt
import numpy as np
import torch
from sf_vae import F0Dectection, PFanalyse, Vowel, TestPFestimation, plot_pitch, plot_f2, PtdbDataset
from sf_vae import VAE, Praat, RVAE
import seaborn as sns

sns.set()


class TestAudioTools(unittest.TestCase):
    def test_Vowel_dataset(self):
        dataset = Vowel()
        out = dataset.get_signal(file_audio="w01ae.wav")
        plt.plot(out['signal'])
        plt.show()

    def test_PFestimation(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        analyse = PFanalyse(model_vae=vae, factor='f0',
                            path_trajectory=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\pitch\f1-400_f2-2000")
        # analyse.H_0_estimation()
        # analyse.analyse(path=r"D:\These\Git\source-filter-vae\OUT.wav", seuil=-120)
        # analyse.analyse(path=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa010c.wav", seuil=-40, plot=True)
        # analyse.analyse(path=r"D:\These\data\Audio\Phonemes\vowel\data\women\w02uw.wav", seuil=-150, plot=True)
        # analyse.analyse(path=r"D:\These\Slides\MDVAE-CSI\audio\pitch_trans_increase_WG.wav", seuil=-150, plot=True)
        # analyse.analyse(path=r"D:\These\data\Audio\Phonemes\Test_Sturnus\w02uw.wav", seuil=-120, plot=True)
        analyse.method_2(path=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa010c.wav",
                         seuil=0.13, plot=True)
        # analyse.invariance_propriety(path_wavs=r"D:\These\Git\source-filter-vae\sf_vae\sf_vae\f0_detection\invariance-proprety")

    def test_PFestimation_Vowel(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        test = TestPFestimation(
            path_trajectory=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\pitch\f1-400_f2-2000",
            path_data=r"D:\These\data\Audio\Phonemes\vowel\data\men",
            # path_data=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\014_f",
            model_vae=vae,
            factor='f0')
        # test.run_f0(seuil=-180)
        test.run_f0_PTDB()
        # test.run_f2(seuil=-300)
        # test.estimation_lambda()
        # test.comparison_methods()
        # test.reverb_robustness(seuil=-300)
        # test.vae_dataset_accuracy(seuil=-300)
        # test.histogram_dataset()
        # test.noise_robustness(seuil=-300, SNR=[None, 40, 20, 30, 10, 0, -5, -10])

    def test_f0estimation(self):
        # praat = Praat(file_name=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\pitch\f1-600_f2-1500\pitch-250.wav")
        # praat = Praat(file_name=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa010c.wav")
        # praat.draw_spectrogram()

        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        f0_tracking = F0Dectection(model_vae=vae)
        # f0_tracking.likelihood_ratio_test()
        # f0_tracking.plotting()
        # f0_tracking.embedding()
        f0_tracking.regression()
        # f0_tracking.analyze(path=r"D:\These\Sites\site-sfvae\demos\sf_vae\controlling\whispering\spec3_withoutf0.wav")
        # f0_tracking(path=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa010c.wav")
        f0_tracking(path=r"D:\These\data\Audio\Phonemes\vowel\synthesis_soundgen\pitch\f1-400_f2-2000\pitch-120.wav")
        f0_tracking(path=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s\01a_f\01aa010c.wav")
        f0_tracking(path=r"D:\These\data\Audio\Phonemes\vowel\data\women\w01oo.wav")
        f0_tracking(path=r"D:\These\Git\source-filter-vae\OUT.wav")

    def test_qualitative_results(self):
        vae = VAE()
        checkpoint = torch.load(r"checkpoints\vae_trained")
        vae.load_state_dict(checkpoint['model_state_dict'])
        # plot_pitch(model_vae=vae)
        plot_f2(model_vae=vae)

    def test_PTDB(self):
        ptdb = PtdbDataset()
        ptdb.h5_creation(dir_save=r'sf_vae/sf_vae/f0_detection/PTDB_TUG-4.hdf5')
