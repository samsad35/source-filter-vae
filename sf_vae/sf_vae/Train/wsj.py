import os
import librosa
import numpy as np
import pandas
from pandas import DataFrame
import h5py
import pickle
from tqdm import tqdm
from ..utils import AudioTools


class WSJ:
    table: DataFrame

    def __init__(self, directory_name: str):
        self.directory_name = directory_name
        self.files = os.listdir(self.directory_name)
        self.files_remove = ['dots', 'dot_spec.doc', 'infos.txt', 'prompts']
        try:
            for file in self.files_remove:
                self.files.remove(file)
        except:
            pass
        self.tools = AudioTools()
        # ---
        self.ext = "wav"

    def generator_id(self):
        for identity in self.files:
            yield identity, self.directory_name + "/" + identity

    def generator_files(self, f):
        files = librosa.util.find_files(f, ext=self.ext)
        for file in files:
            yield file.split('.')[0].split("\\")[-1], file

    def generator(self):
        for id, id_rep in self.generator_id():
            for name, file in self.generator_files(id_rep):
                yield id, name, file

    def get_information(self, index) -> tuple:
        """

        :param index:
        :return:
        """
        id = self.table.iloc[index]['id']
        name = self.table.iloc[index]['name']
        file = self.table.iloc[index]['file_path']
        return id, name, file

    def get_input(self, file):
        signal, rate = self.tools.load(file)
        _, magnitude, _ = self.tools.stft(signal)
        return magnitude ** 2

    def create_table(self):
        files_list = []
        id_list = []
        name_list = []
        with tqdm(total=12776, desc="Create table for WSJ: ") as pbar:
            for id, name, file in self.generator():
                files_list.append(file)
                id_list.append(id)
                name_list.append(name)
                pbar.set_description(f"Create table for WSJ: {name}")
                pbar.update(1)
        self.table = pandas.DataFrame(
            np.array([id_list, name_list, files_list]).transpose(),
            columns=['id', 'name', 'file_path'])

    @staticmethod
    def read_h5(file_h5,
                position: dict = None) -> np.ndarray:
        """

        :param file_h5:
        :param position:
        :return:
        """
        id = position['id']
        name = position['name']
        return np.array(file_h5[f'/{id}/{name}'])

    def arborescence(self, file_h5):
        """

        :param file_h5:
        :return:
        """
        for id, id_rep in self.generator_id():
            file_h5.create_group(id)

    def create_h5(self, dir_save=r'H5',
                  section: str = "train",
                  name_checkpoint: str = "temp_wsj_checkpoint.pkl",
                  restart: bool = False):
        """

        :param section:
        :param dir_save:
        :param name_checkpoint:
        :param restart:
        :return:
        """
        self.create_table()
        path = f"{dir_save}//WSJ_audio_{section}.hdf5"
        if not restart:
            checkpoint = self.read_dict(name_checkpoint)
            print(f"Continue to write a h5 file: checkpoint {checkpoint}")
            index = checkpoint['index']
            totalOfFrames = checkpoint['totalOfFrames']
            totalOfFile = checkpoint['totalOfFile']
            file_h5 = h5py.File(path, 'r+')  # creation of H5 FILE.
            id, name, file = self.get_information(index)
            if file_h5.get(f'/{id}/{name}'):
                index = checkpoint['index'] + 1
            if os.path.isfile('errors.pkl'):
                errors = self.read_dict('errors.pkl')
        else:
            if os.path.isfile(path):
                os.remove(path)
            index = 0
            totalOfFrames = 0
            totalOfFile = 0
            file_h5 = h5py.File(path, 'a')
            print("Create a new H5")
            self.arborescence(file_h5)
            errors = []
        with tqdm(total=self.table.shape[0] - index, desc=f"WSJ's H5 creation (directory: {dir_save})") as pbar:
            while index < self.table.shape[0]:
                try:
                    id, name, file = self.get_information(index)
                    pbar.set_description(f"WSJ's H5 creation (path: {path}) {name}")
                    group_temp = file_h5[f'/{id}']
                    images = self.get_input(file)
                    totalOfFrames += images.shape[0]
                    totalOfFile += 1
                    image_h5 = group_temp.create_dataset(name=name, data=images, dtype="float32")
                    image_h5.attrs.create('id', id)
                    self.save_dict(name_checkpoint,
                                   {'index': index, 'totalOfFrames': totalOfFrames, 'totalOfFile': totalOfFile})
                except:
                    errors.append({'id': id, 'name': name})
                    self.save_dict("errors.pkl", errors)
                index += 1
                pbar.update(1)
        group_info = file_h5.create_group('information')
        group_info.attrs.create("totalOfSegment", totalOfFrames)
        group_info.attrs.create("totalOfFile", totalOfFile)
        file_h5.flush()
        file_h5.close()

    @staticmethod
    def save_dict(name_file, dicA):
        """

        :param name_file:
        :param dicA:
        :return:
        """
        a_file = open(name_file, "wb")
        pickle.dump(dicA, a_file)
        a_file.close()

    @staticmethod
    def read_dict(file='temp.pkl'):
        """

        :param file:
        :return:
        """
        a_file = open(file, "rb")
        output = pickle.load(a_file)
        return output


if __name__ == '__main__':
    wsj = WSJ(directory_name=r"D:\These\data\Audio\WSJ0\wsj0_si_tr_s")
    wsj.create_table()
    print(wsj.table)

