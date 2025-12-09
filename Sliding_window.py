import statistics
import wave
import numpy as np
import matplotlib.pyplot as plt
import keystrokes_detection.xls_handler as xls
import librosa
from scipy.signal import find_peaks


frame_length = 4800    

def sliding_window(audio_array):
    audio_array = abs(audio_array)
    window_data = []   
    sum = 0
    for i in range(0, min(len(audio_array), frame_length)):
        sum += audio_array[i]
    
    window_data.append(sum)
    
    if(len(audio_array)<frame_length): return window_data
    
    for i in range(1, len(audio_array)-frame_length): 
        sum = sum - audio_array[i-1] + audio_array[i+frame_length-1]
        window_data.append(sum) 

    for i in range(len(audio_array)-frame_length, len(audio_array)): 
        sum = sum - audio_array[i-1]
        window_data.append(sum) 
        
    return window_data

def window_peak_detection(window_data, number_of_keys):
    keystrokes = []
    kc = 0
    
    while kc<number_of_keys:
        m = max(window_data)
        index = window_data.index(m)
        keystrokes.append(index)
        for i in range(max(index - frame_length-5,0), min(index + frame_length+5, len(window_data))):
             window_data[i] = 0
        kc+=1
    keystrokes.sort()
    return keystrokes

     
    
def plot_waveform(filename, var, number_of_keys):
    
    audio_array, sample_rate = librosa.load(filename, sr=None)
    sample_duration = 1 / sample_rate
    print(f"One sample lasts for {sample_duration:6f} seconds")
    tot_samples = len(audio_array)
    duration = tot_samples / sample_rate
    print(f"The audio lasts for {duration} seconds")
    time = np.linspace(0, duration, num=len(audio_array))
    win_data = sliding_window(audio_array)
    my_data = [x/30.0 for x in win_data]

    time_intervals  = window_peak_detection(win_data, number_of_keys=number_of_keys )
    keystrokes = np.zeros(len(audio_array))    
    for i in range(len(time_intervals)):
        for j in range(time_intervals[i], min(time_intervals[i]+frame_length,len(audio_array))):
            keystrokes[j]= 0.02
            
    print("keystrokes" ,time_intervals,[win_data[x] for x in time_intervals])
    plt.figure()
    plt.plot(time, audio_array, label='Acoustic Signal')
    plt.plot(time, my_data, color='black', label='Sum of each window frame')
    plt.plot(time, keystrokes, color='r', label='Predicted Keystrokes')    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # plt.title(f'Predicted Keystrokes Plot_{var}_n={number_of_keys}')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    
    return time_intervals      

# a random number generator that gets a seed between 0 and 255, and generates a random number between 0 and FFFFFF
def mapToColor(seed):
    np.random.seed(seed)
    return np.random.randint(0, 0xFFFFFF)   




if __name__ == '__main__':
    for j in range(1, 10):
        for i in range(1, 10):
            print(f"Color for {i} is {mapToColor(i):06X}")
        print('-----------------')
        
    exit()    
       
    # hop_length =   441     # 10 ms
    # frame_length = 4410   # 100 ms
    # number_of_compare = 15
    # plot_waveform('TestFiles\sample2/words.wav', "original_sound", 50)
    avr = []
    for i in range(1,2):
        base_addres = r"TestFiles/1/words"
        wav_filename = f'{base_addres}\\word_{i}.wav'
        xls_filename = f'{base_addres}\\word_{i}.xlsx'
        data = xls.read_excel_data(xls_filename) 
        number_of_keys = len(data)
        for key in data:
            avr.append(key[1] - key[0]) 
        keystroke_avr = np.average(avr)
        keystroke_std = np.std(avr)
        
        print("\nKeystroke average duration=", keystroke_avr, "\nKeystroke Std=" , keystroke_std)
#        frame_length = int((keystroke_avr + keystroke_std) * 44100)
        # frame_length = int((keystroke_avr) * 44100)
        # print("\nFrame Length = ", frame_length)

        time_intervals = plot_waveform(wav_filename, i, number_of_keys)
        # for key in data:
        #     plt.axvspan(key[0], key[1],  color='darkgray')
        
        print("\n\n new:", i)
        for k in range(len(data)-1):
            print("GT+++++>", data[k+1][0]- data[k][0])   
   #     for i in range (len(time_intervals)-1):
   #         print("press to press==>", (time_intervals[i+1][0] - time_intervals[i][0]) / 44100)  # press to press

    input("enter something")
    




