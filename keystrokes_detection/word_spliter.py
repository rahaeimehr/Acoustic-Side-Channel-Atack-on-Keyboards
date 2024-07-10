"""
Written by Alireza Taheri ( March 2022 )

Inspired by Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.
"""
import os
import wave
from copy import deepcopy
import numpy as np
import pandas as pd
import pyaudio
from scipy.io import wavfile as wav
import noisereduce as nr
import xls_handler
from scipy.signal import wiener


class WordSpliter:
    def __init__(self, n_seconds_of_silence=4, remove_from_start_time=1, reduce_noise=True, c_sharp=True,
                 tolerance=None, factor=3.1, file_address='./'):

        self.reduce_noise = reduce_noise
        self.c_sharp = c_sharp
        if self.c_sharp:
            self.filepath = file_address + '/words.wav'
            self.dataset_csv_file_address = file_address + '/words.txt'
        else:
            self.filepath = file_address + '/main.wav'
            self.dataset_csv_file_address = file_address + '/main.csv'

        self.factor = factor
        self.n_seconds_of_silence = n_seconds_of_silence
        self.tolerance = tolerance
        self.output = file_address
        self.remove_from_start_time = remove_from_start_time
        self.sampling_rate = 0
        self.sound_data = []
        self.time = []
        self.wav_read()

    # File input (single WAV file path -> sound data encoded as array)
    def wav_read(self, ):
        """Return 1D NumPy array of wave-formatted audio data denoted by filename.
        Input should be a string containing the path to a wave-formatted audio file.
        """

        self.sampling_rate, self.sound_data = wav.read(self.filepath)
        # print(self.sound_data)
        print("sampling_rate:", self.sampling_rate)
        # print(max(self.sound_data))
        # print(min(self.sound_data))
        self.time = np.arange(len(self.sound_data)) / self.sampling_rate

        if type(self.sound_data[0]) == np.ndarray:
            self.sound_data = self.sound_data[:, 0]

    # Sound preprocessing before keystroke detection
    def silence_threshold(self, ):
        """Return the silence threshold of the sound data.
        The sound data should begin with n-seconds of silence.
        """

        num_samples = self.sampling_rate * self.n_seconds_of_silence
        start_num_samples = int(self.sampling_rate * self.remove_from_start_time)

        if self.reduce_noise:
            noise_clip = self.sound_data[start_num_samples:num_samples + start_num_samples]
            noise_reduced = nr.reduce_noise(y=self.sound_data, sr=self.sampling_rate, y_noise=noise_clip)
            # noise_reduced = wiener(self.sound_data,noise = 0.5).astype(np.int16)
            # noise_reduced = self.myNoiseReducer(self.sound_data,frameLength= 0.02).astype(np.int16)

            # noise_reduced = self.myNoiseReducer(self.sound_data,frameLength= 0.007).astype(np.int16)

            # noise_reduced = self.myNoiseReducer(self.sound_data,frameLength= 0.01).astype(np.int16)

            # Remove silence part from sound
            self.sound_data = noise_reduced[num_samples + start_num_samples:]
            self.save_wave(self.sound_data, 0)  # for saveing the sound without noise
        else:
            self.sound_data = self.sound_data[num_samples + start_num_samples:]

    def myNoiseReducer(self, sound, noise=None, frameLength=0.1):
        frame_Length = int(self.sampling_rate * frameLength)
        sum = 0
        m = 0
        start = 0

        for i in range(len(sound)):
            # sum += abs(sound[i])
            m = max(m, abs(sound[i]))
            if (((i + 1) % frame_Length) == 0):
                # avg = sum / frame_Length
                for j in range(start, i + 1):
                    # sound[j] = avg
                    sound[j] = m
                # sum = 0
                m = 0
                start = i + 1
        return sound
        # TODO: Fixed this function
        for j in range(start, len(sound)):
            sound[j] = 0



    def splitter(self):

        self.silence_threshold()

        if self.c_sharp:
            df_list = self.split_line()
            # print("df_list", df_list)

        else:
            df = pd.read_csv(self.dataset_csv_file_address)
            df_list = df.values.tolist()
            # print(df_list)

        space_number = 1

        sound_start = df_list[0][0] / 2.0

        word_timing = []

        for step in range(0, len(df_list)):

            word_timing.append([df_list[step][0] - sound_start, df_list[step][1] - sound_start, chr(df_list[step][4])])

            # print(step, df_list[step][4], chr(df_list[step][4]))

            if df_list[step][4] == 'Space' or df_list[step][4] == 'Return' or df_list[step][4] == 13:
                # end = df_list[step-1][1] + ((df_list[step][0] - df_list[step-1][1]) / 2) - self.n_seconds_of_silence
                end = (df_list[step - 1][1] + df_list[step][0]) / 2.0
                # end = (df_list[step][1] + df_list[step+1][0])/2

                if step < len(df_list) - 1:
                    start = (df_list[step][1] + df_list[step + 1][0]) / 2.0

                self.save_wave(self.sound_data[int(sound_start * self.sampling_rate): int(end * self.sampling_rate)], space_number)
                word_timing.pop() # for removing spaces or inters
                xls_handler.write_matrix_to_file(word_timing, f'{self.output}/words/word_{space_number}.xlsx',
                                                 column_titles=["down", "up", "key"])
                word_timing = []

                sound_start = start
                space_number += 1
                last_sep = step

        if last_sep < len(df_list) - 1:
            xls_handler.write_matrix_to_file(word_timing, f'{self.output}/words/word_{space_number}.xls',
                                             column_titles=["down", "up", "key"])
            self.save_wave(self.sound_data[int(end * self.sampling_rate):], space_number)

    def save_wave(self, sound_data, number):
        # print(number, sound_data, len(sound_data))
        # chunk = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        p = pyaudio.PyAudio()

        if self.c_sharp:
            RATE = 44100
            width = p.get_sample_size(FORMAT)
        else:
            RATE = 44100
            width = p.get_sample_size(FORMAT)
        os.makedirs(f'{self.output}/words', exist_ok=True)
        wf = wave.open(f'{self.output}/words/word_{number}.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(width)
        wf.setframerate(RATE)
        sound_data_int16 = np.asarray(sound_data, dtype=np.int16)
        wf.writeframes(sound_data_int16.tobytes())
        wf.close()

    def get_key_events_from_data(self, no_of_key_press, no_of_key_release):
        results = []
        # time of the detected event
        time = 1.3
        # the probability of the event type (Release or Press)
        probability = 0.6
        results.append((time, probability))
        return results

    def split_line(self, split_char=','):
        # print(self.dataset_csv_file_address)
        lst = list()
        with open(self.dataset_csv_file_address) as file:
            for line in file:
                line = line.strip()
                elements = line.split(split_char)

                elements[0] = int(elements[0]) / 10000000 - self.n_seconds_of_silence - self.remove_from_start_time
                elements[2] = int(elements[2], 16)
                elements[3] = int(elements[3], 16)
                lst.append(elements)

        assert lst[0][0] > 0, "The length of the initial silence, faulty noises, or both is not correct!"

        my_list = []

        for num in range(int(len(lst) / 2)):
            # [press_time, release_time, key]
            # print(num, [lst[2 * num][0], lst[2 * num + 1][0], 0, 0, lst[num][2]])
            my_list.append([lst[2 * num][0], lst[2 * num + 1][0], 0, 0, lst[2 * num][2]])
        return my_list

# # file_address = r"E:\alireza\augusta\codes\net5\dataset\638120949256235517590"
# file_address = r'E:\alireza\augusta\codes\recorder\dataset\638138995564246274257/'
# # file_address = r"E:\alireza\augusta\codes\keylogger\keylogger\datasets\3"
#
#
# n_seconds_of_silence = 5
# remove_from_start_time = 1
# factor = 1
# c_sharp = True
#
#
# wave_spliter = WordSpliter(n_seconds_of_silence=n_seconds_of_silence,
#                            file_address=file_address, c_sharp=c_sharp,
#                            remove_from_start_time=remove_from_start_time, factor=factor)
# wave_spliter.splitter()
