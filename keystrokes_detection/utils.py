import numpy as np
import wave
from copy import deepcopy
import pandas as pd
import pyaudio
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav


class WordPredictor:
    def __init__(self, file_address='./'):

        self.num_plot_cols = 3
        self.tolerance = None
        self.factor = 0.9
        self.dataset_csv_file_address = file_address + '/main.csv'
        self.filepath = file_address
        self.output = file_address
        self.sampling_rate = 44100
        self.sound_data = []
        self.time = []
        self.wav_read()

    # File input (single WAV file path -> sound data encoded as array)
    def wav_read(self, ):
        """Return 1D NumPy array of wave-formatted audio data denoted by filename.
        Input should be a string containing the path to a wave-formatted audio file.
        """
        self.sampling_rate, self.sound_data = wav.read(self.filepath)
        self.time = np.arange(len(self.sound_data)) / self.sampling_rate

        if type(self.sound_data[0]) == np.ndarray:
            self.sound_data = self.sound_data[:, 0]

    @property
    def keystroke_duration(self):
        return 0.2

    def calc_threshold(self, ):
        return max(np.amax(self.sound_data), abs(np.amin(self.sound_data))) * self.factor

    def detect_keystrokes(self, ):
        print("detect_keystrokes")

        threshold = self.calc_threshold()
        len_sample = int(self.sampling_rate * self.keystroke_duration)
        time_of_keystrokes = []
        keystrokes = []
        i = 0
        while i < len(self.sound_data):
            if abs(self.sound_data[i]) > threshold:
                a, b = i, i + len_sample
                if b > len(self.sound_data):
                    b = len(self.sound_data) - 1

                while abs(self.sound_data[b]) > threshold and b > a:
                    b -= 1
                keystroke = self.sound_data[a:b]

                trailing_zeros = np.array([0 for _ in range(len_sample - (b - a))])
                keystroke = np.concatenate((keystroke, trailing_zeros))
                keystrokes.append(keystroke)
                time_of_keystrokes.append(self.time[a:b])
                i = b - 1
            i += 1
        return len(keystrokes), np.array(keystrokes), np.array(time_of_keystrokes)

    def visualize_keystrokes(self):
        print("visualize_keystrokes")
        """Display each keystroke detected in WAV file specified by filepath."""
        # self.sound_data = self.wav_read()
        keystrokes, time_of_keystrokes = self.detect_keystrokes()
        n = len(keystrokes)
        print(f'Number of keystrokes detected in "{self.filepath}": {n}')
        print('Drawing keystrokes...')
        num_cols = self.num_plot_cols
        num_rows = n / num_cols + 1
        plt.figure(figsize=(num_cols * 6, num_rows * .75))
        for i in range(n):

            plt.subplot(int(num_rows), num_cols, i + 1)
            plt.title(f'Index: {i}')
            plt.plot(keystrokes[i])
            print(f"\n\nDetected keystroke number: {i + 1}")
            print(f"Start at: {time_of_keystrokes[i][0]} \nEnd at: {time_of_keystrokes[i][-1]}")
            print(f"Duration of detected keystroke:{time_of_keystrokes[i][-1] - time_of_keystrokes[i][0]}")

        plt.show()

    @property
    def calc_probability(self):
        return 100

    def get_key_events_from_data(self, no_of_key_press, no_of_key_release):

        match = None
        data = []
        times = []
        while not match:
            self.factor = self.factor - 0.10
            print(self.factor)
            number_of_keys , data , times = self.detect_keystrokes()
            if number_of_keys == no_of_key_press:
                match = 'ok'
                print("match", number_of_keys)

        results = [] #{'press_time':1, 'release_time': 2, 'probability'}
        for key in range(len(data)):
            print(times[key][0], times[key][-1], self.calc_probability)
            results.append((times[key][0], times[key][-1], self.calc_probability))
        return results



f = r'E:\alireza\augusta\codes\keylogger\keylogger\datasets\13\word_1.wav'
run = WordPredictor(file_address=f).get_key_events_from_data(7,7)
# run = WordPredictor(file_address=f).visualize_keystrokes()
print(run)
