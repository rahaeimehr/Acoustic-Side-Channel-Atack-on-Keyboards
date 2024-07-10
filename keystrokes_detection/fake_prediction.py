import librosa
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import soundfile as sf
from scipy.signal import find_peaks, argrelextrema

from predictor import KeystrokesPredictor
from graph_maker import PredictedTree


class KeyDdetector:
    def __init__(self, file_path, model_address='',
                 analyzed=True, delta_time_min_factor=0.5, delta_time_max_factor=1.5):

        self.analyzed = analyzed
        self.file_path = file_path
        self.delta_time_max_factor = delta_time_max_factor
        self.delta_time_min_factor = delta_time_min_factor
        self.inter_keystroke_intervals = []
        print("Running ... ")
        self.tp = KeystrokesPredictor(model_address)
        self.predicted_keys_per_steps = []
        self.text_analyser(self.file_path)

    def split_line(self, filename, split_char=','):
        print(filename)
        lst = list()
        with open(filename) as file:
            for line in file:
                line = line.strip()
                elements = line.split(split_char)
                # print(elements)
                elements[0] = int(elements[0]) / 10000000  # Time
                elements[2] = chr(int(elements[2], 16))  # ascii (hex)
                elements[3] = int(elements[3], 16)  # scan code
                # print("elements:", elements)
                lst.append(elements)
        return lst

    def hex_to_ascii(self, dataset_csv_file_address):
        keystrokes = []
        up_times = []
        down_time = []
        data = []
        df_list = self.split_line(dataset_csv_file_address)
        for i in range(int((len(df_list)) / 2)):
            keystrokes.append(df_list[2 * i][2])
            up_times.append(df_list[2 * i + 1][0])
            down_time.append(df_list[2 * i][0])

        for i in range(len(keystrokes) - 1):
            delta = ((down_time[i + 1] + up_times[i + 1]) / 2) - ((down_time[i] + up_times[i]) / 2)
            data.append([keystrokes[i], keystrokes[i + 1], delta])
        return data

    def text_analyser(self, file_address):
        dataset_csv_file_address = file_address + r"/words.txt"
        data = self.hex_to_ascii(dataset_csv_file_address)
        print(data)
        word_number = 1
        desired_word = 4
        for i in range(len(data)):

            if data[i][0] == '\r':

                word_number += 1

            elif word_number == desired_word:
                self.inter_keystroke_intervals.append(data[i][2])
        self.inter_keystroke_intervals.pop(-1)
        print(self.inter_keystroke_intervals)
        self.key_detector()
        self.make_graph_of_predictions()

    def key_detector(self, ):

        for delta_time in self.inter_keystroke_intervals:
            delta_time_min = delta_time * self.delta_time_min_factor
            delta_time_max = delta_time * self.delta_time_max_factor
            finder = self.tp.keystrokes_finder(delta_time_min, delta_time_max, analyzed=self.analyzed)
            self.predicted_keys_per_steps.append(finder)

    def make_graph_of_predictions(self):
        PredictedTree(self.predicted_keys_per_steps)


model_address = r'C:\Users\r_rah\source\repos\rahaeimehr\Recorder\bin\Debug\net6.0-windows\dataset\638246711393010642538'
file = model_address + r'/words.txt'

plot = True
analyzed = False
delta_range = 0.05
delta_time_min_factor = 1 - delta_range
delta_time_max_factor = 1 + delta_range

KeyDdetector(file_path=model_address, analyzed=analyzed,
             delta_time_min_factor=delta_time_min_factor, delta_time_max_factor=delta_time_max_factor,
             model_address=model_address)
