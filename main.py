import wave
import numpy as np
import matplotlib.pyplot as plt
import keystrokes_detection.xls_handler as xls
import librosa
import os
import time
from audio_handler import AudioHandler

def find_segments(bool_array, time):
        segments = []
        start = None
        for i in range(len(bool_array)):
            if bool_array[i] and start is None:
                start = i
            elif not bool_array[i] and start is not None:
                segments.append([start, i - 1])
                start = None
        if start is not None:
            segments.append([start, len(bool_array) - 1])
        return segments
    
def find_best_threshold(energy, time, best_threshold, factor, num_desired_keystrokes):
    num_keystrokes = 0
    old_best_thr = []
    segments = None
    max_found_keystrokes = 0

    while num_keystrokes != num_desired_keystrokes:

        threshold = np.percentile(energy, best_threshold)
        keys = np.where(energy > threshold, 1, 0)

        segments = find_segments(keys, time)
        num_keystrokes = len(segments)
        print(num_keystrokes)
        max_found_keystrokes = num_keystrokes if num_keystrokes >= max_found_keystrokes else max_found_keystrokes
        old_best_thr.append(best_threshold)
        print(old_best_thr)
        if num_keystrokes > num_desired_keystrokes:
            best_threshold = float(best_threshold + factor)
        elif num_keystrokes < num_desired_keystrokes:
            best_threshold = float(best_threshold - factor)

        if best_threshold in old_best_thr:
            factor = float(factor * 0.1)

        if best_threshold == 0:
            print("max_found_keystrokes", max_found_keystrokes)
            raise RuntimeError
    print("final_best_threshold", best_threshold)
    print("_threshold", threshold)
    # return segments, threshold
    

    return segments, keys * max(energy), threshold

        
def energy_analyser(audio_array, sample_rate):

    hop_length =  1
    frame_length = sample_rate // 10 

    energy = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    frames = range(len(energy))
    time = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
   
    return time, energy
            
def amplitude_envelope(audio_array, sample_rate):
       
        hop_length =  44100
        frame_length = 44100
        amplitude_envelope = []
        
        # calculate amplitude envelope for each frame
        for i in range(0, len(audio_array), hop_length): 
            amplitude_envelope_current_frame = max(audio_array[i: i + frame_length]) 
            amplitude_envelope.append(amplitude_envelope_current_frame)  

        amp_env = np.array(amplitude_envelope) 
        frames = range(len(amp_env))
        time = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
        return time, amp_env
    
def plot_waveform(filename, i, number_of_keys):
    
    audio_array, sample_rate = librosa.load(filename)
    # duration in seconds of 1 sample
    sample_duration = 1 / sample_rate
    print(f"One sample lasts for {sample_duration:6f} seconds")
    
    # duration of debussy audio in seconds
    tot_samples = len(audio_array)
    duration = tot_samples / sample_rate 
    print(f"The audio length:  {duration} seconds")

    time = np.linspace(0, duration, num=len(audio_array))
    # ae_time , amp_env = amplitude_envelope(audio_array, sample_rate)
    etime, energy = energy_analyser(audio_array, sample_rate)
    plt.figure()
    #plt.plot(time, audio_array)
    plt.plot(etime,energy)
    plt.show(block=False)

    segments, keys, threshold = find_best_threshold(smooth(energy,500), etime, best_threshold=50, factor=1, num_desired_keystrokes=number_of_keys)



    # plt.plot(ae_time, amp_env, color='y', label='Amplitude Envelope')   
    plt.plot(etime, energy, color='r', label='Energy (RMSE)')    
    plt.step(etime, keys, color='darkmagenta', where='post') 
    plt.plot(etime, np.ones(len(etime)) * threshold, color='y', label='Threshold')   
    plt.plot(etime, smooth(energy,300), 'black', lw=2)     
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform Plot_{i}_n={number_of_keys}')
    plt.grid(True)
    plt.show(block=False)
    
def plot_waveform(au:AudioHandler):
    
    time = np.linspace(0, au.audio_length, num=au.sample_count)
    plt.plot(time, au.left_channel)
    plt.show(block=False)

            
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == '__main__':

    start_time = time.perf_counter()

    base_folder = os.getcwd() + r"/TestFiles/1/"        
    au = AudioHandler(base_folder + 'words.wav')
    au.print_audio_info()
    
    plot_waveform(au)
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

    # for i in range(1,4):
    #     base_addres = 'TestFiles/2/words'
    #     wav_filename = f'{base_addres}/word_{i}.wav'
    #     xls_filename = f'{base_addres}/word_{i}.xlsx'
    #     data = xls.read_excel_data(xls_filename) 
    #     number_of_keys = len(data)
    #     plot_waveform(wav_filename, i, number_of_keys)
    #     first_time = data[0][0] / 2
    #     for key in data:
    #         plt.axvspan(key[0], key[1],  color='darkgray')    
    # input("enter something")












