import librosa
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import soundfile as sf
from scipy.signal import find_peaks, argrelextrema
from scipy.io import wavfile as wav

from predictor import KeystrokesPredictor
from graph_maker import PredictedTree


class KeyDdetector:
    def __init__(self, file_path, output_dir="./keystrokes", model_address='', num_desired_keystrokes=None,
                 plot=True, analyzed=True, delta_time_min_factor=0.5, delta_time_max_factor=1.5):

        self.frame_length = None
        self.hop_length = None
        self.sr = None
        self.analyzed = analyzed
        self.file_path = file_path
        self.output_dir = output_dir
        self.plot = plot
        self.delta_time_max_factor = delta_time_max_factor
        self.delta_time_min_factor = delta_time_min_factor
        self.inter_keystroke_intervals = []
        self.keystroke_times = []
        self.num_desired_keystrokes = num_desired_keystrokes
        self.output_words = []
        # print("Running ... ")
        self.tp = KeystrokesPredictor(model_address)
        self.predicted_keys_per_steps = []
        self.window_analyser()
        



    def key_detector(self, ):
        for delta_time in self.inter_keystroke_intervals:
            delta_time_min = delta_time * self.delta_time_min_factor
            delta_time_max = delta_time * self.delta_time_max_factor
            finder = self.tp.keystrokes_finder(delta_time_min, delta_time_max, analyzed=self.analyzed)
            print(len(finder))
            self.predicted_keys_per_steps.append(finder)

    def make_graph_of_predictions(self):
        self.output_words = PredictedTree(self.predicted_keys_per_steps, len(self.keystroke_times)).my_prediction(self.num_desired_keystrokes)
        return self.output_words


    def key_detection(self):
        
        for i in range(len(self.keystroke_times)-1):
            delta = ( (self.keystroke_times[i+1][0] - self.keystroke_times[i][0]) / self.sr)
            self.inter_keystroke_intervals.append(delta)
            
        for i in range(len(self.keystroke_times)):
            keystroke_segment = self.signal[self.keystroke_times[i][0]: self.keystroke_times[i][1]]
            # Save the current keystroke as a WAV file
            os.makedirs(self.output_dir, exist_ok=True)
            filename = os.path.join(self.output_dir, "keystroke_{}.wav".format(i + 1))
            sf.write(filename, keystroke_segment, self.sr)

            # if self.plot:
            #     plt.figure()
            #     plt.plot(keystroke_segment)
            #     plt.title("Keystroke {} Waveform".format(i + 1))
            #     plt.xlabel("Time (samples)")
            #     plt.ylabel("Amplitude")
            #     plt.show(block=True)

        print("inter_keystroke_intervals:", self.inter_keystroke_intervals)
        print("len inter_keystroke_intervals:", len(self.inter_keystroke_intervals))

        self.key_detector()
        self.make_graph_of_predictions()
        return self.output_words
    
    def sliding_window(self, audio_array):
        audio_array = abs(audio_array)
        window_data = []   
        sum = 0
        for i in range(0, min(len(audio_array), self.frame_length)):
            sum += audio_array[i]
        
        window_data.append(sum)
        
        if(len(audio_array)<self.frame_length): return window_data
        
        for i in range(1, len(audio_array)-self.frame_length): 
            sum = sum - audio_array[i-1] + audio_array[i+self.frame_length-1]
            window_data.append(sum) 

        for i in range(len(audio_array)-self.frame_length, len(audio_array)): 
            sum = sum - audio_array[i-1]
            window_data.append(sum) 
            
        return window_data

    def window_peak_detection(self, window_data, number_of_keys):
        keystrokes = []
        kc = 0
        
        while kc<number_of_keys:
            m = max(window_data)
            index = window_data.index(m)
            keystrokes.append(index)
            for i in range(max(index - self.frame_length-5,0), min(index + self.frame_length+5, len(window_data))):
                window_data[i] = 0
            kc+=1
        keystrokes.sort()
        return keystrokes


    def window_analyser(self):


        self.frame_length = 4410   # 100ms
        number_of_compare = 5
        
        self.signal, self.sr = librosa.load(self.file_path, sr=None)
        sample_duration = 1 / self.sr
        # print(f"One sample lasts for {sample_duration:6f} seconds")
        tot_samples = len(self.signal)
        duration = 1 / self.sr * tot_samples
        # print(f"The audio lasts for {duration} seconds")

        time = np.linspace(0, duration, num=len(self.signal))
       
        
        win_data = self.sliding_window(self.signal)
        # my_data = [x/30.0 for x in win_data]     
 
        
        if self.plot:    
            plt.figure()
            plt.plot(time, self.signal)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Waveform Plot')
            plt.grid(True)
            plt.show(block=True)
            
        if not self.num_desired_keystrokes:
            self.num_desired_keystrokes = int(input("Enter number of Keystrokes: "))

            if isinstance(self.num_desired_keystrokes, int):
                print("You entered:", self.num_desired_keystrokes)
            else:
                print("The value is not an integer.")


        time_intervals  = self.window_peak_detection(win_data, number_of_keys=self.num_desired_keystrokes )
        keystrokes = np.zeros(len(self.signal))    
        for i in range(len(time_intervals)):
            for j in range(time_intervals[i], min(time_intervals[i]+self.frame_length,len(self.signal))):
                keystrokes[j]= 0.02
                
            self.keystroke_times.append([time_intervals[i], min(time_intervals[i]+self.frame_length,len(self.signal))])
            
            
            
        if self.plot:    
            plt.figure()
            plt.plot(time, self.signal)
            plt.plot(time, keystrokes, color='black', label='Amplitude Envelope')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Waveform Plot')
            plt.grid(True)
            plt.show(block=True)
            
        # print(self.keystroke_times)