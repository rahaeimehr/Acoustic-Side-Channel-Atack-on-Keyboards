"""
Written by Alireza Taheri ( March 2022 )

Inspired by Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.
"""
import os
from copy import deepcopy
import librosa
import numpy as np
from scipy.io import wavfile as wav
from predictor import KeystrokesPredictor
from graph_maker import PredictedTree
import logging
import scipy.signal as signal
from matplotlib import pyplot as plt
import soundfile as sf


logging.basicConfig(handlers=[
    # logging.FileHandler("log.txt"),
    logging.StreamHandler()
], level=logging.INFO)


class KeystrokesCounter:
    def __init__(self, filepath,
                 n_seconds_of_silence=4,
                 remove_from_start_time=1,
                 tolerance=None,
                 keystroke_duration=0.3,
                 num_plot_cols=2,
                 factor=3,
                 output=True,
                 delta_time_min_factor=0.9,
                 delta_time_max_factor=1.1,
                 number_of_press=1):

        self.sound_data_copy = None
        self.number_of_press = number_of_press
        self.filepath = filepath
        self.n_seconds_of_silence = n_seconds_of_silence
        self.tolerance = tolerance
        self.output = output
        self.keystroke_duration = keystroke_duration
        self.num_plot_cols = num_plot_cols
        self.remove_from_start_time = remove_from_start_time
        self.factor = factor
        self.sampling_rate = 0
        self.sound_data = []
        self.time = []

        self.delta_time_min_factor = delta_time_min_factor
        self.delta_time_max_factor = delta_time_max_factor
        self.wav_read()
        self.tp = KeystrokesPredictor()
        self.predicted_keys_per_steps = []

    # File input (single WAV file path -> sound data encoded as array)
    def wav_read(self, ):
        """Return 1D NumPy array of wave-formatted audio data denoted by filename.

        Input should be a string containing the path to a wave-formatted audio file.
        """
        self.sampling_rate, self.sound_data = wav.read(self.filepath)
        self.time = np.arange(len(self.sound_data)) / self.sampling_rate

        if type(self.sound_data[0]) == np.ndarray:
            self.sound_data = self.sound_data[:, 0]

        self.sound_data_copy = deepcopy(self.sound_data)
    # Sound preprocessing before keystroke detection
    def silence_threshold(self, ):
        """Return the silence threshold of the sound data.

        The sound data should begin with n-seconds of silence.
        """
        if self.n_seconds_of_silence != 0:
            num_samples = self.sampling_rate * self.n_seconds_of_silence
            start_num_samples = self.sampling_rate * self.remove_from_start_time
            silence = self.sound_data_copy[start_num_samples:num_samples]
            measured = np.std(silence)
            logging.info(f"measured std:{measured}")
            if not self.tolerance:
                self.tolerance = measured + (measured * 0.03)
            # Remove silence part from sound
            self.sound_data = self.sound_data_copy[num_samples:]

            return max(np.amax(silence), abs(np.amin(silence))) * self.factor

    def remove_random_noise(self, threshold=None):
        """Return a copy of sound_data where random noise is replaced with 0's.

        The original sound_data is not mutated.
        """
        threshold = threshold  # or self.silence_threshold()
        sound_data_copy = deepcopy(self.sound_data)
        for i in range(len(sound_data_copy)):
            if abs(sound_data_copy[i]) < threshold:
                sound_data_copy[i] = 0
        return sound_data_copy

    # Keystroke detection: (encoded array -> all keystroke data in array)
    def detect_keystrokes(self, ):
        print("detect_keystrokes")
        """Return slices of sound_data that denote each keystroke present.

        Returned keystrokes are coerced to be the same length by appending trailing
        zeros.

        Current algorithm:
        - Calculate the "silence threshold" of sound_data.
        - Traverse sound_data until silence threshold is exceeded.
        - Once threshold is exceeded, mark that index as "a".
        - Identify the index 0.3s ahead of "a", and mark that index as "b".
        - If "b" happens to either:
              1) denote a value that value exceeds the silence threshold (aka:
                 likely impeded on the next keystroke waveform)
              2) exceed the length of sound_data
          then backtrack "b" until either:
              1) it denotes a value lower than the threshold
              2) "b" is 1 greater than "a"
        - Slice sound_data from index "a" to "b", and append that slice to the list
          to be returned. If "b" was backtracked, then pad the slice with trailing
          zeros to make it 0.3s long.

        :type sound_file  -- NumPy array denoting input sound clip
        :type sample_rate -- integer denoting sample rate (samples per second)
        :rtype            -- NumPy array of NumPy arrays
        """
        threshold = self.silence_threshold()
        print("Sound Threshold:", threshold)
        # sig = self.remove_random_noise(threshold)  # todo
        # self.sound_data = sig

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
        # return np.array(keystrokes), np.array(time_of_keystrokes)

        return len(keystrokes), np.array(keystrokes), np.array(time_of_keystrokes)

    # Display detected keystrokes (WAV file -> all keystroke graphs)
    def visualize_keystrokes(self):
        print("visualize_keystrokes")
        """Display each keystroke detected in WAV file specified by filepath."""
        # self.sound_data = self.wav_read()
        _, keystrokes, time_of_keystrokes = self.get_key_events_from_data(self.number_of_press)
        n = len(keystrokes)
        logging.info(f'Number of keystrokes detected in "{self.filepath}": {n}')
        logging.info('Drawing keystrokes...')
        num_cols = self.num_plot_cols
        num_rows = n / num_cols + 1
        plt.figure(figsize=(num_cols * 6, num_rows * .75))
        for i in range(n):

            plt.subplot(int(num_rows), num_cols, i + 1)
            plt.title(f'Index: {i}')
            plt.plot(keystrokes[i])
            logging.info(f"\n\nDetected keystroke number: {i + 1}")
            logging.info(f"Start at: {time_of_keystrokes[i][0]} \nEnd at: {time_of_keystrokes[i][-1]}")
            logging.info(f"Duration of detected keystroke:{time_of_keystrokes[i][-1] - time_of_keystrokes[i][0]}")
            if i + 1 < n:
                delta_time_for_predict = time_of_keystrokes[i + 1][0] - time_of_keystrokes[i][-1]
                logging.info(f"Time interval until the next keystroke: {delta_time_for_predict}")
                delta_time_min = delta_time_for_predict * self.delta_time_min_factor
                delta_time_max = delta_time_for_predict * self.delta_time_max_factor
                logging.info(f"delta_time_for_predict:{delta_time_min, delta_time_max}")
                self.predicted_keys_per_steps.append(self.key_detector(delta_time_min, delta_time_max))
            else:
                logging.info(f"Time interval until the end of file: {self.time[-1] - time_of_keystrokes[i][-1]}")
        for i in self.predicted_keys_per_steps:
            print(i)
        logging.info(f"predicted_keys_per_steps {self.predicted_keys_per_steps}")
        self.make_graph_of_predictions()
        plt.show()

    def key_detector(self, delta_time_min, delta_time_max):
        finder = self.tp.keystrokes_finder(delta_time_min, delta_time_max)
        # logging.info(f"Predicted key: {finder}")
        return finder

    def make_graph_of_predictions(self):
        # logging.info("help:")
        # for i in range(len(self.predicted_keys_per_steps[0][0])):
        #     logging.info([self.predicted_keys_per_steps[0][0][i], self.predicted_keys_per_steps[0][1][i]])

        PredictedTree(self.predicted_keys_per_steps)

    def get_key_events_from_data(self, no_of_key_press):

        match = None
        data = []
        times = []
        while not match:
            print("factor:", self.factor)
            number_of_keys, data, times = self.detect_keystrokes()
            if number_of_keys == no_of_key_press:
                match = 'ok'
                print("match", number_of_keys)
            self.factor = self.factor - 0.10

        return number_of_keys, data, times

        # results = [] #{'press_time':1, 'release_time': 2, 'probability'}
        # for key in range(len(data)):
        #     print(times[key][0], times[key][-1], "100")
        #     results.append((times[key][0], times[key][-1], "100"))
        # return results


wav_path_k = r"E:\alireza\augusta\codes\net5\dataset\638120949256235517590/word_3.wav"
n_seconds_of_silence = 0
remove_from_start_time = 1
tolerance = None
keystroke_duration = 0.2
num_plot_cols = 2
factor = 1
delta_time_min_factor = 0.3
delta_time_max_factor = 1.1
number_of_press = 3


wav_path_k = r"E:\alireza\augusta\codes\net5\dataset\638120949256235517590/words.wav"
n_seconds_of_silence = 5
remove_from_start_time = 1
tolerance = None
keystroke_duration = 0.2
num_plot_cols = 2
factor = 1
delta_time_min_factor = 0.3
delta_time_max_factor = 1.1
number_of_press = 15

key_counter = KeystrokesCounter(filepath=wav_path_k, n_seconds_of_silence=n_seconds_of_silence,
                                remove_from_start_time=remove_from_start_time, tolerance=tolerance,
                                keystroke_duration=keystroke_duration, num_plot_cols=num_plot_cols, factor=factor,
                                delta_time_min_factor=delta_time_min_factor,
                                delta_time_max_factor=delta_time_max_factor, number_of_press=number_of_press)
key_counter.visualize_keystrokes()
