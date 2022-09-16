from librosa.filters import mel as librosa_mel_fn
import torch
from .stft_waveGlow import STFT
import soundfile as sf
import librosa
import numpy as np
import sounddevice as sd
import pyworld as pw
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa.display as display


class AudioTools:
    def __init__(self):
        mel_basis = librosa_mel_fn(22050, 1024, 80, 0.0, 8000.0)
        self.mel_basis = torch.from_numpy(mel_basis).float()
        self.stft_waveglow = STFT(filter_length=1024, hop_length=256, win_length=1024, window='hann')
        # WaveGlow
        self.waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.eval()
        self.waveglow.to(torch.device('cuda'))

    @staticmethod
    def load(path_wav: str, resample=None, return_f0: bool = False):
        sig, rate = sf.read(path_wav)
        if resample is not None:
            sig = librosa.resample(sig, rate, resample)
            rate = resample
        sig = sig / np.max(np.abs(sig))
        # signal_audio, index = librosa.effects.trim(sig, top_db=30)
        # signal_audio = np.pad(signal_audio, int(1024 // 2), mode='reflect')
        if return_f0:
            f0, _, _ = pw.wav2world(sig, rate)
            return sig, rate, f0
        else:
            return sig, rate

    def stft(self, y):
        y = torch.from_numpy(y).float()
        y = y.unsqueeze(0)
        assert (y.min() >= -1)
        assert (y.max() <= 1)
        magnitudes, phases = self.stft_waveglow.transform(y)  # Same STFT as used in WaveGlow
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        return mel_output[0], magnitudes[0], phases[0]

    def spec2mel(self, spec):
        magnitudes = spec
        magnitudes = torch.from_numpy(magnitudes)
        mel_output = torch.matmul(self.mel_basis, magnitudes.float())
        return mel_output

    def istft(self, spectrogram, phase):
        signal = self.stft_waveglow.inverse(spectrogram, phase)[0, 0, :]
        return signal

    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-5):
        """
        PARAMS
        ------
        C: compression factor
        """
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def audio_reconstruction_waveGlow(self, mel_spectrogram: torch.Tensor):
        '''
            WaveGlow
        :param mel_spectrogram:
        :param spectrogram:
        :return:
        '''
        audio_tot = torch.tensor([]).to(mel_spectrogram.device)

        # CUDA's memory management :
        with torch.no_grad():
            audio = self.waveglow.infer(mel_spectrogram[None, :, :].to(mel_spectrogram.device))
            audio_tot = torch.cat((audio_tot, audio), dim=1)

        audio_numpy = audio_tot[0].data.cpu().numpy()
        audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))
        return audio_numpy

    @staticmethod
    def write(signal, name: str = "temp.wav"):
        write(name, 16000, signal)

    @staticmethod
    def play(signal):
        sd.play(signal, 16000)

    @staticmethod
    def plot_spectrogram(spec, show=True, save: str = None):
        fig, ax = plt.subplots()
        # img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time')
        img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=16000, y_axis='linear',
                               x_axis='time', hop_length=256)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        if save is not None:
            plt.savefig(save)
        plt.show()

    @staticmethod
    def griffin_lim(S, **kwargs):
        signal = librosa.griffinlim(S, hop_length=256, win_length=1024, **kwargs)
        return signal
