import unittest
from src import AudioTools


class TestAudioTools(unittest.TestCase):
    def test_load(self):
        audio = AudioTools()
        signal, rate = audio.load(path_wav=r"D:\These\data\Audio\LJSpeech-1.1\wavs\LJ001-0015.wav", resample=16000)
        self.assertEqual(rate, 16000)
        self.assertEqual(len(signal.shape), 1)


