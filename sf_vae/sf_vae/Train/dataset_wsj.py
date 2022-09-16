from torch.utils.data import Dataset
import h5py
import numpy as np
from .wsj import WSJ
import torch
import random
import warnings
warnings.filterwarnings("ignore")


class WSJDataset(Dataset):
    def __init__(self, directory_name: str = None, section: str = None, h5_path: str = None):
        self.wsj = WSJ(directory_name)
        print(f"Create table: {section}", end="")
        self.wsj.create_table()
        print(f" ... ok")
        self.h5 = h5py.File(f'{h5_path}\\WSJ_audio_{section}.hdf5', 'r')
        # -----
        self.index_wav = 0
        self.wav_list = np.arange(0, len(self.wsj.table))
        self.number_frames = 0
        self.current_frame = 0
        self.shuffle = True
        random.shuffle(list(self.wav_list))

    def read_h5(self, position: dict = None) -> np.ndarray:
        emotion = position['emotion']
        id = position['id']
        level = position['level']
        name = position['name']
        return np.array(self.h5[f'/{id}/{emotion}/{level}/{name}'])

    def __len__(self):
        return np.array(self.h5[f'/information'].attrs['totalOfSegment'])

    def reset(self):
        if self.index_wav >= len(self.wav_list):
            self.index_wav = 0
            if self.shuffle:
                random.shuffle(list(self.wav_list))

    def __getitem__(self, item):
        if self.current_frame == self.number_frames:
            while True:
                self.reset()
                id, name, file = self.wsj.get_information(self.wav_list[self.index_wav])
                info = dict(id=id, name=name)
                self.spectrogram = self.wsj.read_h5(self.h5, position=info).transpose()
                self.index_wav += 1
                self.current_frame = 0
                self.number_frames = self.spectrogram.shape[0]
                if self.number_frames > 0:
                    break
        self.s = self.spectrogram[self.current_frame, :]
        self.current_frame += 1
        return torch.from_numpy(self.s).type(torch.FloatTensor)
