import librosa
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import soundfile as sf
from predictor import KeystrokesPredictor
from graph_maker import PredictedTree


class KeyDdetector():
    def __init__(self, file_path, output_dir="./keystrokes", num_desired_keystrokes=32, keystroke_duration=0.32,
                         plot=False, delta_time_min_factor=0.1, delta_time_max_factor=1.8):

        self.file_path = file_path
        self.output_dir = output_dir
        self.num_desired_keystrokes = num_desired_keystrokes
        self.keystroke_duration = keystroke_duration
        self.plot = plot
        self.delta_time_max_factor = delta_time_max_factor
        self.delta_time_min_factor = delta_time_min_factor
        self.inter_keystroke_intervals = []
        print("Running ... ")
        self.tp = KeystrokesPredictor()
        self.predicted_keys_per_steps = []

    def keystroke_finder(self,):

        # Load your audio data into a numpy array, e.g. using librosa:
        y, sr = librosa.load(self.file_path, sr=None)

        # S1:
        # Define a bandpass filter to isolate the frequency range of interest (e.g. 2000-5000 Hz)
        fs = sr
        # print(sr)
        f1, f2 = 2000, 5000


        # S2:
        # Calculate the power spectral density (PSD) of the input signal
        # f, psd = signal.welch(y, fs=sr, nperseg=2048, window='hamming', nfft=4096)
        # # Find the dominant frequency range in the PSD (excluding low frequencies)
        # idx = np.argmax(psd[1:]) + 1
        # f1 = f[idx]
        # f2 = 2 * f1

        print("Frequency range of interest: {:.0f}-{:.0f} Hz".format(f1, f2))
        # Compute the energy of the audio signal

        w = [f1 / (sr / 2), f2 / (sr / 2)]

        b, a = signal.butter(4, Wn=w, btype='bandpass', analog=False)

        # Apply the filter to the audio signal
        filtered_y = signal.filtfilt(b, a, y)

        # Find the best threshold to match the desired number of keystrokes
        keypress_indices = []
        num_keystrokes = 0
        best_threshold = 0.5
        factor = 0.01
        old_best_thr = []
        while num_keystrokes != self.num_desired_keystrokes:

            threshold = best_threshold * np.max(filtered_y)
            keypress_indices = np.where(filtered_y > threshold)[0]
            num_keystrokes = len(keypress_indices)

            old_best_thr.append(best_threshold)

            if num_keystrokes > self.num_desired_keystrokes:
                best_threshold += factor
            elif num_keystrokes < self.num_desired_keystrokes:
                best_threshold -= factor

            # print("best_threshold", best_threshold)
            if best_threshold in old_best_thr:
                factor = factor * 0.1
                # print("new factor:", factor)

        # Print the detected keypress indices
        print("Detected keypress indices:", keypress_indices)
        print("number of pressed keys:", len(keypress_indices))

        # Convert the keypress indices to time units based on the sampling rate of the audio
        keypress_times = (keypress_indices / sr).astype(float)
        print("keypress_times", keypress_times)

        # Compute the inter-keystroke intervals
        self.inter_keystroke_intervals = np.diff(keypress_times)
        print("inter_keystroke_intervals:", self.inter_keystroke_intervals)

        # Split the audio signal into segments corresponding to individual keystrokes
        keystroke_segments = []
        for i in range(len(keypress_indices)):
            start_index = int(round(keypress_indices[i] - self.keystroke_duration / 2 * sr))
            end_index = int(round(keypress_indices[i] + self.keystroke_duration / 2 * sr))
            keystroke_segments.append(y[start_index:end_index])

        # Concatenate the keystroke segments into a single array for further analysis
        keystroke_array = np.concatenate(keystroke_segments)

        if self.plot:
            # Plot the waveform of the keystroke array
            plt.figure()
            plt.plot(keystroke_array)
            plt.title("Waveform of Keystrokes")
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.show()

        # Create a directory to save the keystroke files in
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Split the audio signal into segments corresponding to individual keystrokes
        keystroke_segments = []
        for i in range(len(keypress_indices)):
            start_index = int(round(keypress_indices[i] - self.keystroke_duration / 2 * sr))
            end_index = int(round(keypress_indices[i] + self.keystroke_duration / 2 * sr))
            keystroke_segment = y[start_index:end_index]
            keystroke_segments.append(keystroke_segment)

            # Save the current keystroke as a WAV file
            filename = os.path.join(self.output_dir, "keystroke_{}.wav".format(i + 1))
            sf.write(filename, keystroke_segment, sr)

            if self.plot:
                # Plot the waveform of the current keystroke
                plt.figure()
                plt.plot(keystroke_segment)
                plt.title("Keystroke {} Waveform".format(i + 1))
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.show()

        self.key_detector()
        self.make_graph_of_predictions()

        return 1

    def key_detector(self,):

        for delta_time in self.inter_keystroke_intervals:
            delta_time_min = delta_time * self.delta_time_min_factor
            delta_time_max = delta_time * self.delta_time_max_factor
            finder = self.tp.keystrokes_finder(delta_time_min, delta_time_max)
            self.predicted_keys_per_steps.append(finder)

    def make_graph_of_predictions(self):
        PredictedTree(self.predicted_keys_per_steps)


# r'E:\alireza\augusta\codes/keyboard2.wav'
file = r'E:\alireza\augusta\codes\net5\dataset\638120949256235517590/words.wav'

detector = KeyDdetector(file_path=file, output_dir="../keystrokes_detection/keystrokes_1", num_desired_keystrokes=15, ).keystroke_finder()



