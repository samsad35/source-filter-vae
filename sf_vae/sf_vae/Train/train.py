import torch
import time
from torch.utils.data import DataLoader, Dataset
from .loss import Loss
from .follow_up import Follow
from ..model import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ..utils import AudioTools


class Train:
    def __init__(self, model: VAE,
                 training_data: Dataset,
                 validation_data: Dataset,
                 config_training: dict = None):
        self.config_training = config_training
        self.model = model
        self.batch_size = self.config_training['batch_size']
        self.epochs = self.config_training['epochs']
        self.device = torch.device(self.config_training['device'])
        self.learning_rate = self.config_training['learning_rate']
        self.load_epoch = 0
        self.start = time.time()
        self.tools = AudioTools()

        # To device:
        self.model = self.model.to(self.device)

        # Dataloader:
        self.training_dataloader = DataLoader(training_data, batch_size=self.batch_size)
        self.validation_dataloader = DataLoader(validation_data, batch_size=self.batch_size)

        # Optimizer:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config_training['scheduler'][1],
        #                                                  gamma=self.config_training['scheduler'][2])
        # Loss :
        self.loss = Loss()

        # Follow up, save parameters and early-stopping:
        self.follow_up = Follow("VAE", self.epochs, dir_save=r'checkpoints/VAE')
        self.parameters = dict(model=None, optimizer=None, epoch=None, loss=None, time=None)
        self.training_data = training_data

    def train_process(self, epoch: int = 1):
        self.model.train()
        loss_epoch = 0
        one_shot = False
        for x in self.training_dataloader:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            x_, mu, logvar, z = self.model(x)
            if epoch % 1 == 0 and not one_shot:
                signal = self.tools.griffin_lim(np.sqrt(torch.transpose(x_[0:50], 0, 1).cpu().detach().numpy()))
                self.tools.write(signal, name=f"out.wav")
                signal = self.tools.griffin_lim(np.sqrt(torch.transpose(x[0:50], 0, 1).cpu().detach().numpy()))
                self.tools.write(signal, name=f"out_reel.wav")
                one_shot = True
            loss = self.loss(x, x_, mu, logvar, batch_size=self.batch_size, beta=1.0)
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        loss_epoch /= len(self.training_dataloader.dataset)
        return loss_epoch

    def validation_process(self):
        self.model.eval()
        loss_epoch = 0
        for x in self.validation_dataloader:
            x = x.to(self.device)
            x_, mu, logvar, z = self.model(x)
            loss = self.loss(x, x_, mu, logvar, beta=1.0, batch_size=self.batch_size)
            loss_epoch += loss.item()
        loss_epoch /= len(self.validation_dataloader.dataset)
        return loss_epoch

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(f"{path}\\model_checkpoint")
        checkpoint_ = torch.load(f"{path}\\model")
        self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.follow_up.load_dict(path)
        self.follow_up.best_loss = checkpoint_['loss']
        self.start = checkpoint['time']
        self.load_epoch = checkpoint['epoch']
        print(f"model:ok  | optimizer:{optimizer}  |  loss: {self.follow_up.best_loss}  |  epoch: {self.load_epoch}]")

    def __call__(self):
        with tqdm(total=self.epochs, desc="Training: ") as pbar:
            for epoch in range(self.load_epoch + 1, self.epochs + 1):
                loss_train = self.train_process(epoch=epoch - 1)
                loss_validation = self.validation_process()
                pbar.set_description(f"Training: epoch [{epoch}/{self.epochs}]; loss train = {loss_train};"
                                     f" loss validation = {loss_validation}")
                pbar.update(1)
                finish = time.time()
                self.parameters = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                                       epoch=epoch, loss=loss_validation, time=finish)
                self.follow_up(epoch, loss_train, loss_validation, (finish - self.start) / 60, self.parameters,
                               print_tabulate=False)



