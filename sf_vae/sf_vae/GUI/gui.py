import sys
import os
import tkinter
from tkinter import *
import matplotlib.pyplot as plt
import torch
from tkinter.filedialog import *
from tkinter import messagebox
import numpy as np
from ..utils import AudioTools
from ..utils import Praat
from ..method import Controlling
import sounddevice as sd
from scipy.io.wavfile import write
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Interface:
    z: np.ndarray
    time: np.ndarray
    signal: np.ndarray
    signal_recons: np.ndarray
    switch_formant: StringVar
    modify_bool: IntVar
    switch_variable: StringVar
    list_formant: list
    list_time: list
    z_time: list

    def __init__(self, model: torch.nn.Module = None, device: str = None, path: str = None):
        self.master = Tk()
        self.master.title("sf-vae")
        self.master.geometry("1200x800")
        self.master.wm_iconbitmap(r'sf_vae\sf_vae\Gui\icon.ico')
        self.master.resizable(width=False, height=False)

        # Model VAE + device:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Audio tools:
        self.tools = AudioTools()
        self.praat_num = IntVar()

        # Controlling:
        self.list_time = [0]
        self.list_formant = []
        self.list_x = [220]
        self.list_y = [356]
        self.cwd = os.getcwd()
        self.control = Controlling(path=path, model=model, device=device)

        # Buttons/ canvas ...:
        Button(self.master, text="File .wav", command=self.get_signal).place(relx=0.5, rely=0.1, anchor=CENTER)
        Scale(self.master, orient='horizontal', from_=2, to=6, resolution=1, tickinterval=25, length=100,
              label='Praat', variable=self.praat_num).place(relx=0.1, rely=0.1, anchor=CENTER)
        self.canvas2 = Canvas(self.master, height=400, width=1400, cursor='crosshair', relief=RAISED)
        self.canvas2.bind('<ButtonRelease>', self.do)

    def get_signal(self):
        self.path_wav = askopenfilename(title="Open an audio file",
                                   filetypes=[('wav files', '.wav'), ('all files', '.*')])
        self.signal, fs = self.tools.load(self.path_wav, resample=16000)
        write("OUT.wav", fs, self.signal)
        self.z = self.control.get_z(self.path_wav)
        self.time = np.linspace(0, self.signal.shape[0], self.signal.shape[0]) / fs
        Button(self.master, text="Play", command=self.play).place(relx=0.2, rely=0.1, anchor=CENTER)
        Button(self.master, text="Go", command=self.run).place(relx=0.5, rely=0.95, anchor=CENTER)
        fig = Figure(figsize=(14, 2))
        a = fig.add_subplot(111)
        a.plot(self.time, self.signal, color='blue')
        a.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().place(relx=0.5, rely=0.25, anchor=CENTER)
        canvas.draw()
        praat = Praat(file_name=f'{self.cwd}\\OUT.wav', num_formant=int(self.praat_num.get()))
        praat.draw_formant_pitch(save=True, draw=False)
        self.canvas2.place(relx=0.5, rely=0.65, anchor=CENTER)
        img = PhotoImage(file=f"{self.cwd}\\spectrogram_pitch.png")
        self.master.one = img
        self.canvas2.create_image(0, 0, anchor=NW, image=img)
        self.reset()

        self.menu()
        self.modify_bool = IntVar()
        self.modify_bool.set(0)
        Checkbutton(self.master, text="Suppress the formant", variable=self.modify_bool).place(relx=0.3, rely=0.43,
                                                                                               anchor=CENTER)

    def menu(self):
        self.switch_variable = StringVar(value="off")
        Radiobutton(self.master, text="swipe off", variable=self.switch_variable,
                    indicatoron=False, value="off", width=8).place(relx=0.1, rely=0.43, anchor=CENTER)
        Radiobutton(self.master, text="On", variable=self.switch_variable,
                    indicatoron=False, value="on", width=8).place(relx=0.16, rely=0.43, anchor=CENTER)

        self.switch_formant = StringVar(value="f0")
        Radiobutton(self.master, text="F1", variable=self.switch_formant,
                    indicatoron=False, value="f1", width=8).place(relx=0.5, rely=0.43, anchor=CENTER)
        Radiobutton(self.master, text="F2", variable=self.switch_formant,
                    indicatoron=False, value="f2", width=8).place(relx=0.56, rely=0.43, anchor=CENTER)
        Radiobutton(self.master, text="F0", variable=self.switch_formant,
                    indicatoron=False, value="f0", width=8).place(relx=0.62, rely=0.43, anchor=CENTER)
        Radiobutton(self.master, text="F3", variable=self.switch_formant,
                    indicatoron=False, value="f3", width=8).place(relx=0.68, rely=0.43, anchor=CENTER)
        self.canvas2.create_line(1220, 75, 1220, 328, width=2, fill='red')
        self.canvas2.create_line(1215, 328, 1225, 328, width=2, fill='red')
        self.canvas2.create_line(1215, 75, 1225, 75, width=2, fill='red')

    def play(self):
        sd.play(self.signal, 16000)

    def play_(self):
        sd.play(self.signal_recons, 16000)

    def run(self):
        alpha = np.array([])
        if not self.list_formant:
            z_final = self.z
        else:
            for i in range(len(self.z_time)):
                if i == 0:
                    if self.switch_formant.get() == 'f1':
                        alpha = np.concatenate((alpha, np.linspace(300, self.list_formant[i], self.z_time[i])))
                    elif self.switch_formant.get() == 'f2':
                        alpha = np.concatenate((alpha, np.linspace(1005, self.list_formant[i], self.z_time[i])))
                    elif self.switch_formant.get() == 'f3':
                        alpha = np.concatenate((alpha, np.linspace(2000, self.list_formant[i], self.z_time[i])))
                    elif self.switch_formant.get() == 'f0':
                        alpha = np.concatenate((alpha, np.linspace(100, self.list_formant[i], self.z_time[i])))
                else:
                    alpha = np.concatenate(
                        (alpha, np.linspace(self.list_formant[i - 1], self.list_formant[i], self.z_time[i] - self.z_time[i - 1])))
            alpha = np.concatenate(
                (alpha, np.linspace(self.list_formant[-1], self.list_formant[-1], self.z.shape[0] - alpha.shape[0])))
        self.control(path_wav=self.path_wav, y=alpha, factor=self.switch_formant.get(), path_new_wav="out.wav")
        praat = Praat(file_name=f'{self.cwd}\\out.wav', num_formant=int(self.praat_num.get()))
        praat.draw_formant_pitch(save=False, draw=True)
        self.signal_recons, sr = self.tools.load(path_wav="out.wav", resample=16000)
        Button(self.master, text="Play", command=self.play_).place(relx=0.9, rely=0.95, anchor=CENTER)

    def reset(self):
        self.canvas2.delete("all")
        img = PhotoImage(file=f"{self.cwd}\\spectrogram_pitch.png")
        self.master.one = img
        self.canvas2.create_image(0, 0, anchor=NW, image=img)
        self.canvas2.create_image(0, 0, anchor=NW, image=img)
        self.canvas2.create_line(1220, 75, 1220, 328, width=2, fill='red')
        self.canvas2.create_line(1215, 328, 1225, 328, width=2, fill='red')
        self.canvas2.create_line(1215, 75, 1225, 75, width=2, fill='red')
        self.list_time = [0]
        self.list_formant = []
        self.z_time = []
        self.list_x = [220]
        self.list_y = [356]

    def do(self, event):
        if self.switch_variable.get() == "off":
            self.reset()

        if self.switch_formant.get() == 'f1':
            self.canvas2.create_text(1240, 75, text="1000")
            self.canvas2.create_text(1240, 328, text="300")
        elif self.switch_formant.get() == 'f2':
            self.canvas2.create_text(1240, 75, text="2800")
            self.canvas2.create_text(1240, 328, text="1000")
        elif self.switch_formant.get() == 'f0':
            self.canvas2.create_text(1240, 75, text="300")
            self.canvas2.create_text(1240, 328, text="100")
        elif self.switch_formant.get() == 'f3':
            self.canvas2.create_text(1240, 75, text="3200")
            self.canvas2.create_text(1240, 328, text="2400")

        if self.switch_variable.get() == 'on':
            t = (event.x - 224) / (987 / self.time[-1])
            f1 = int((437 - event.y) / 0.36)
            f2 = int((467 - event.y) / 0.14)
            f0 = int((438 - event.y) / 1.21)
            f3 = int((1074 - event.y) / 0.311)
            if t > self.time[-1] or t < 0 or f1 > 1000 or f1 < 300 or t < self.list_time[-1]:
                messagebox.showerror("Error", "Please select the good f1/f2 value and time")
            else:
                self.canvas2.create_line(event.x - 10, event.y, event.x + 10, event.y, width=2, fill='red')
                self.canvas2.create_line(event.x, event.y - 10, event.x, event.y + 10, width=2, fill='red')
                self.canvas2.create_line(self.list_x[-1], self.list_y[-1], event.x, event.y, width=2, fill='yellow')
                self.list_x.append(event.x)
                self.list_y.append(event.y)
                if self.switch_formant.get() == 'f1':
                    self.list_formant.append(f1)
                elif self.switch_formant.get() == 'f2':
                    self.list_formant.append(f2)
                elif self.switch_formant.get() == 'f3':
                    self.list_formant.append(f3)
                elif self.switch_formant.get() == 'f0':
                    self.list_formant.append(f0)
                self.list_time.append(t)
                self.z_time.append(int(62 * t))

