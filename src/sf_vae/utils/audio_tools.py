from librosa.filters import mel as librosa_mel_fn
import torch
from .stft_waveGlow import STFT
import soundfile as sf
import librosa
import numpy as np


class AudioTools:
    def __init__(self):
        mel_basis = librosa_mel_fn(22050, 1024, 80, 0.0, 8000.0)
        self.mel_basis = torch.from_numpy(mel_basis).float()
        self.stft_waveglow = STFT(filter_length=1024, hop_length=256, win_length=1024, window='hann')

    def load(self, path_wav: str, resample=None):
        sig, rate = sf.read(path_wav)
        if resample is not None:
            sig = librosa.resample(sig, rate, resample)
            rate = resample
        sig = sig / np.max(np.abs(sig))
        signal_audio, index = librosa.effects.trim(sig, top_db=30)
        signal_audio = np.pad(signal_audio, int(1024 // 2), mode='reflect')
        return signal_audio, rate

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
