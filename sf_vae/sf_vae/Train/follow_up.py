from datetime import datetime
import shutil
import os
import pandas
import torch
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
import string
import random
# from torch.utils.tensorboard import SummaryWriter


class Follow:
    name_dir: str

    def __init__(self, name: str, epochs: int, dir_save: str = ""):
        self.name = name
        self.code = "".join([random.choice(string.ascii_letters) for i in range(5)])
        self.datatime_start = datetime.today()
        self.dir_save = dir_save
        self.create_directory()
        self.table = {"epoch": [], "loss_train": [], "loss_validation": [], "time": []}
        self.best_loss = 1e8
        self.epochs = epochs
        # self.writer = SummaryWriter(comment=r'checkpoint')

    def create_directory(self):
        to_day = "Y" + str(self.datatime_start.date().year) + 'M' + str(self.datatime_start.date().month) + \
                 'D' + str(self.datatime_start.date().day)
        time = str(self.datatime_start.time().hour) + 'h' + str(self.datatime_start.time().minute)
        os.mkdir(f"{self.dir_save}\\{self.name}-{to_day}-{time}")
        self.name_dir = f"{self.name}-{to_day}-{time}"

    def find_best_model(self, loss_validation):
        if loss_validation <= self.best_loss:
            self.best_loss = loss_validation
            return True
        else:
            return False

    def save_model(self, boolean: bool, parameters: dict):
        torch.save(parameters, f'{self.dir_save}\\{self.name_dir}\\model_checkpoint')
        if boolean:
            torch.save(parameters, f'{self.dir_save}\\{self.name_dir}\\model')
            # print(f"Model saved: [loss:{parameters['loss']}]")

    def save_plot(self):
        plt.plot(self.table['loss_train'])
        plt.plot(self.table['loss_validation'])
        plt.xlabel('epoch')
        plt.ylabel('loss train')
        plt.savefig(f'{self.dir_save}\\{self.name_dir}\\plot.jpeg')
        plt.close()

    def push(self, epoch: int, loss_train: float, loss_validation: float, time: float):
        self.table['epoch'].append(epoch)
        self.table['loss_train'].append(loss_train)
        self.table['loss_validation'].append(loss_validation)
        self.table['time'].append(time)

    def save_csv(self):
        df = pandas.DataFrame(self.table)
        df.to_csv(path_or_buf=f'{self.dir_save}\\{self.name_dir}\\mdvae_table.csv')

    def save_dict(self):
        a_file = open(f"{self.dir_save}\\{self.name_dir}\\table.pkl", "wb")
        pickle.dump(self.table, a_file)
        a_file.close()

    def load_dict(self, path: str):
        a_file = open(f"{path}\\table.pkl", "rb")
        self.table = pickle.load(a_file)

    def __call__(self, epoch: int, loss_train: float,
                 loss_validation: float, time: float, parameters: dict, print_tabulate: bool = True):
        if print_tabulate:
            print()
            print(tabulate([['Epoch', epoch], ['Loss train', round(loss_train, 2)],
                            ['Loss Validation', round(loss_validation, 2)], ['Time', round(time, 2)]],
                           tablefmt="fancy_grid"), '\n')
        bool_best_model = self.find_best_model(loss_validation)
        self.save_model(bool_best_model, parameters)
        self.push(epoch,  round(loss_train, 2), round(loss_validation, 2), round(time, 2))
        self.save_plot()
        self.save_csv()
        self.save_dict()

