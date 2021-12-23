import unittest
from src import AudioTools


class TestAudioTools(unittest.TestCase):
    def test_load(self):
        audio = AudioTools()
        signal, rate = audio.load(path_wav=r"D:\These\data\Audio\LJSpeech-1.1\wavs\LJ001-0015.wav", resample=16000)
        self.assertEqual(rate, 16000)
        # audio.play(signal)
        self.assertEqual(len(signal.shape), 1)

    def test_stft(self):
        audio = AudioTools()
        signal, rate = audio.load(path_wav=r"D:\These\data\Audio\LJSpeech-1.1\wavs\LJ001-0015.wav", resample=16000)
        mel, spec, phase = audio.stft(signal)
        self.assertEqual(spec.shape[0], 513)

    def test_istft(self):
        audio = AudioTools()
        signal, rate = audio.load(path_wav=r"D:\These\data\Audio\LJSpeech-1.1\wavs\LJ001-0015.wav", resample=16000)
        mel, spec, phase = audio.stft(signal)
        signal_ = audio.istft(spec, phase)
        # audio.play(signal_)
        self.assertEqual(signal_.shape, signal.shape)


if __name__ == '__main__':
    unittest.main()


