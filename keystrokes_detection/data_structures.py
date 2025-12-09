import librosa as wav
import numpy as np
import matplotlib.pyplot as plt

class AudioFile:
    audio_path: str
    keystroke_data_path: str
    keystrokes: list 
    
    def __init__(self, audio_path, keystroke_data_path):
        self.audio_path = audio_path
        self.keystroke_data_path = keystroke_data_path
        
        self.keystrokes = list()
        with open(self.keystroke_data_path) as file:
            for line in file:
                line = line.strip()
                elements = line.split(',')
                elements[0] = int(elements[0]) / 10000000     # Time
                elements[2] = chr(int(elements[2], 16))       # ascii (hex)
                elements[3] = int(elements[3], 16)            # scan code
                self.keystrokes.append(elements)
        temp = []
        res = []

        for array in self.keystrokes:
            if array[1] == '0':
                temp.append(array)
            else:
                # Find the last array in temp with the same third field
                flag = False
                for i in reversed(range(len(temp))):
                    if temp[i][2] == array[2]:
                        res.append([temp[i][0], array[0], temp[i][2]])
                        del temp[i]
                        flag = True
                        break      
                if not flag:
                    res.append([0, array[0], array[2]])     
        if len(temp) > 0:
            for i in range(len(temp)):
                res.append([temp[i][0], math.inf, temp[i][2]]) 
        #sorted(res, key=lambda x: x[0])
        self.keystrokes = sorted(res)
        
        self.sound_data , self.sampling_rate = wav.load(self.audio_path, sr=None , mono = False)  
        if len(self.sound_data.shape) > 1:
            self.sound_data = self.sound_data[0 , : ]    
        
        print("\n\nThe content of the file: ", self.keystroke_data_path , "\n\n", self.keystrokes, "\n\n", "-"*50)
        
    def get_nth_keystrokePressTime(self, n):
        if n < len(self.keystrokes):
            return self.keystrokes[n][0]
        else:
            raise IndexError("Keystroke index out of range")

    def get_nth_keystrokeReleaseTime(self, n):
        if n < len(self.keystrokes):
            return self.keystrokes[n][1]
        else:
            raise IndexError("Keystroke index out of range")

    def get_nth_keystrokeChar(self, n):
        if n < len(self.keystrokes):
            return self.keystrokes[n][2]
        else:
            raise IndexError("Keystroke index out of range")
    def trim_sound_data(self, start_time, end_time):
        """
        Trims the sound data to the specified start and end times.
        """
        start_sample = int(start_time * self.sampling_rate)
        end_sample = int(end_time * self.sampling_rate)
        self.sound_data = self.sound_data[start_sample:end_sample]
        


    def plot_keystrokes_on_waveform(self):
        times = np.arange(len(self.sound_data)) / self.sampling_rate
       # plt.figure(figsize=(15,5))
        h  = 0.8 * max(self.sound_data)
        plt.plot(times, self.sound_data, alpha=0.6)
        for press_time, release_time, char in self.keystrokes:
            plt.axvline(press_time, color='green', linestyle='--', label='press')
            plt.axvline(release_time, color='red', linestyle='--', label='release')
            plt.text((press_time + release_time)/2, h, char, color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(self.audio_path.split('/')[-1] )
        plt.show()
