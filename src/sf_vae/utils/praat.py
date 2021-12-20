import parselmouth
from parselmouth.praat import call
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import os
import scipy as sp


# sns.set() # Use seaborn's default style to make attractive graphs
# plt.rcParams['figure.dpi'] = 160 # Show nicely large images in this notebook

class Praat:
    def __init__(self, sr=16000, file_name='', num_formant=3, time_step=None):
        self.sr = sr
        self.num_formant = num_formant
        self.snd = parselmouth.Sound(file_name)
        self.formant = self.snd.to_formant_burg(max_number_of_formants=num_formant, maximum_formant=5500.0,
                                                time_step=time_step)

    def plot_signal_audio(self):
        plt.figure()
        plt.plot(self.snd.xs(), self.snd.values.T)
        plt.xlim([self.snd.xmin, self.snd.xmax])
        plt.xlabel("time [s]")
        plt.ylabel("amplitude")
        plt.show()  # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

    def draw_spectrogram(self, dynamic_range=70, draw=True):
        spectrogram = self.snd.to_spectrogram(window_length=0.015, maximum_frequency=5000)  # 0.03
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        # plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='binary')
        # plt.figure(figsize=(14, 4))
        plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='magma')  # magma
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.xlabel("time [s]")
        plt.ylabel("frequency [Hz]")
        if draw:
            plt.show()

    def draw_intensity(self):
        intensity = self.snd.to_intensity()
        self.draw_spectrogram(draw=False)
        plt.twinx()
        plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
        plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
        plt.grid(False)
        plt.ylim(0)
        plt.ylabel("intensity [dB]")
        plt.show()

    def draw_pitch(self, draw=True, save=False):
        pitch_values, pitch = self.get_pitch()
        pitch_values[pitch_values == 0] = np.nan
        plt.figure(figsize=(14, 4))
        self.draw_spectrogram(draw=False)
        plt.twinx()
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(80, 320)
        plt.ylabel("fundamental frequency [Hz]")
        if save:
            plt.savefig('spectrogram_pitch.png')
        if draw:
            plt.show()

    def pre_emphasize(self):
        self.snd.pre_emphasize()

    def draw_formant(self, draw=True, save=False):
        formant1_tab, formant2_tab, time = self.get_formants()
        plt.figure(figsize=(14, 4))
        self.draw_spectrogram(draw=False)
        plt.twinx()
        plt.plot(time, formant1_tab, 'o', markersize=5, color='w')
        plt.plot(time, formant1_tab, 'o', markersize=2, color='r')
        plt.plot(time, formant2_tab, 'o', markersize=5, color='w')
        plt.plot(time, formant2_tab, 'o', markersize=2, color='g')
        plt.grid(False)
        plt.ylim(0, 5000)
        plt.ylabel("Formant frequency [Hz]")
        if save:
            plt.savefig('spectrogram_pitch.png')
        if draw:
            plt.show()

    def draw_formant_pitch(self, draw=True, save=False):
        pitch_values, pitch = self.get_pitch()
        pitch_values[pitch_values == 0] = np.nan
        plt.figure(figsize=(14, 4))
        self.draw_spectrogram(draw=False)
        formant1_tab, formant2_tab, _, time = self.get_formants()
        # plt.twinx()
        plt.plot(time, formant1_tab, 'o', markersize=5, color='w')
        plt.plot(time, formant1_tab, 'o', markersize=2, color='r')
        plt.plot(time, formant2_tab, 'o', markersize=5, color='w')
        plt.plot(time, formant2_tab, 'o', markersize=2, color='g')
        plt.twinx()
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2, color='b')
        plt.ylim(80, 320)
        plt.ylabel("fundamental frequency [Hz]")
        if save:
            plt.savefig('spectrogram_pitch.png')
        if draw:
            plt.show()

    def get_pitch(self, time_step=None, median_filter=False):
        # Extract selected pitch contour, and
        # replace unvoiced samples by NaN to not plot
        pitch = self.snd.to_pitch_ac(time_step=time_step)
        pitch_values = pitch.selected_array['frequency']
        # pitch_values = [v for v in pitch_values if str(v) != 'nan']
        if median_filter:
            pitch_values = sp.signal.medfilt(pitch_values, 41)
        return pitch_values, pitch

    def get_formants(self, median_filter=False):
        # time = np.arange(0, len(self.snd) / self.sr, 100 * 1 / self.sr)
        time = np.arange(0, len(self.snd) / self.sr, 160 * 1 / self.sr)
        formant1_tab = []
        formant2_tab = []
        formant3_tab = []
        for t in time:
            f1 = self.formant.get_value_at_time(1, t)
            f2 = self.formant.get_value_at_time(2, t)
            f3 = self.formant.get_value_at_time(3, t)
            formant1_tab.append(f1)
            formant2_tab.append(f2)
            formant3_tab.append(f3)

        if median_filter:
            formant1_tab = sp.signal.medfilt(formant1_tab, 49)
            formant2_tab = sp.signal.medfilt(formant2_tab, 49)
            formant3_tab = sp.signal.medfilt(formant3_tab, 49)

        # formant1_tab = [f1 for f1 in formant1_tab if str(f1) != 'nan']
        # formant2_tab = [f2 for f2 in formant2_tab if str(f2) != 'nan']
        # formant3_tab = [f3 for f3 in formant3_tab if str(f3) != 'nan']
        return formant1_tab, formant2_tab, formant3_tab, time

    # This function measures formants using Formant Position formula
    def measureFormants(self):
        # pitch = call(self.snd, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        pointProcess = call(self.snd, "To PointProcess (periodic, cc)", 75, 600)

        formants = call(self.snd, "To Formant (burg)", 0.0025, self.num_formant, 5000, 0.025, 50)
        numPoints = call(pointProcess, "Get number of points")

        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []

        # Measure formants only at glottal pulses
        for point in range(0, numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)

        f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
        f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
        f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']

        # calculate mean formants across pulses
        f1_mean = statistics.mean(f1_list)
        f2_mean = statistics.mean(f2_list)
        # f3_mean = statistics.mean(f3_list)
        #
        # # calculate median formants across pulses, this is what is used in all subsequent calcualtions
        # # you can use mean if you want, just edit the code in the boxes below to replace median with mean
        # f1_median = statistics.median(f1_list)
        # f2_median = statistics.median(f2_list)
        # f3_median = statistics.median(f3_list)

        # return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median
        return f1_list, f2_list, f3_list


# formant = snd.to_formant_burg(max_number_of_formants=3.0,maximum_formant=5500.0)
# print(formant)
#
# print(formant.get_value_at_time(1, 0.2))
# ======================================================================================================================
############################
#       Test : main        #
############################
if __name__ == '__main__':
    path = r'D:\These\data\WSJ0\wsj0_si_tr_s'
    directory = os.listdir(path)
    pitch_total_male = np.array([])
    pitch_total_female = np.array([])
    f1 = np.array([])
    f2 = np.array([])
    try:
        for i, dir in enumerate(directory[:100]):
            print(i)
            files_wav = os.listdir(path + "/" + dir)
            for file in files_wav:
                prat = Praat(file_name=path + "/" + dir + "/" + file, num_formant=5)
                pitch_values, _ = prat.get_pitch()
                formant1_tab, formant2_tab, _ = prat.get_formants()
                formant1_tab = list(filter(lambda num: ~np.isnan(num), formant1_tab))
                formant2_tab = list(filter(lambda num: ~np.isnan(num), formant2_tab))
                f1 = np.concatenate((f1, (formant1_tab)), axis=None)
                f2 = np.concatenate((f2, (formant2_tab)), axis=None)
                pitch_values = list(filter(lambda num: num != 0, pitch_values))

                if '_m' in dir:
                    pitch_total_male = np.concatenate((pitch_total_male, np.mean(pitch_values)), axis=None)
                else:
                    pitch_total_female = np.concatenate((pitch_total_female, np.mean(pitch_values)), axis=None)

    except:
        pass
    # n_bins = 50
    # plt.hist(pitch_total_male, bins=n_bins, color= "skyblue", ec="skyblue")
    # plt.hist(pitch_total_female, bins=n_bins, color='pink', ec="pink")
    # plt.xlabel('Pitch value')
    formant = list([f1, f2])
    pitch = list([pitch_total_male, pitch_total_female])
    fig, ax = plt.subplots()
    # print(formant)
    # ax1.boxplot(pitch, 0, '', patch_artist=True)
    ax.violinplot(formant)
    ax.set_xticklabels(["F1", "F2", ])
    plt.show()
    fig2, ax2 = plt.subplots()
    ax2.violinplot(pitch)
    ax2.set_xticklabels(["male", "female", ])
    plt.show()
