from statistics import mean
import noisereduce as nr
import librosa
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import soundfile as sf


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
    max_signal = np.max(energy)
    segments = None

    while num_keystrokes != num_desired_keystrokes:

        threshold = best_threshold * max_signal
        keys = energy > threshold
        segments = find_segments(keys, time)
        num_keystrokes = len(segments)

        old_best_thr.append(best_threshold)

        if num_keystrokes > num_desired_keystrokes:
            best_threshold = best_threshold + factor
        elif num_keystrokes < num_desired_keystrokes:
            best_threshold = best_threshold - factor

        if best_threshold in old_best_thr:
            factor = factor * 0.1

    print("final_best_threshold", best_threshold)
    print("_threshold", threshold)
    return segments, threshold


def key_detection(segments):
    all_max = []
    inter_keystroke_intervals = []
    print("Number of detected keys:", len(segments))
    for i in range(len(segments)):
        keystroke_segment = signal[segments[i][0]:segments[i][1]]
        # print(f"max of key {i +1} is {max(keystroke_segment)}")
        ind = np.where(keystroke_segment == max(keystroke_segment))[0]
        all_max.append(segments[i][0] + ind)

        # print(f"index is {ind}")
        # print(f"sig {signal[segments[i][0] + ind]}")
        # print(f"time {time[segments[i][0] + ind]}")

        if i < len(segments) - 1:

            keystroke_segment_next = signal[segments[i + 1][0]:segments[i + 1][1]]
            ind_next = np.where(keystroke_segment_next == max(keystroke_segment_next))[0]
            delta = (time[segments[i+1][0] + ind_next] - time[segments[i][0] + ind])
            inter_keystroke_intervals.append(delta[0])

    print("inter_keystroke_intervals:", inter_keystroke_intervals)
    print("len inter_keystroke_intervals:", len(inter_keystroke_intervals))
    return inter_keystroke_intervals, all_max


def energy_analyser(file_path, n_seconds_of_silence, remove_from_start_time, num_desired_keystrokes=False):
    signal, sr = librosa.load(file_path, sr=None)
    noise_clip = signal[remove_from_start_time * sr:n_seconds_of_silence * sr]
    reduced_noised = nr.reduce_noise(signal, sr, y_noise=noise_clip, stationary=True)
    signal = reduced_noised[n_seconds_of_silence * sr:]

    print("sound duration:  ", librosa.get_duration(y=signal, sr=sr))
    sound_duration = librosa.get_duration(y=signal, sr=sr)
    hop_length = 1
    frame_length = 1024
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    frames = range(len(energy))
    time = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    s_time = np.linspace(0, len(signal) / sr, num=len(signal))
    return time, energy, s_time, signal, sound_duration


def split_line(filename, split_char=','):
    print(filename)
    lst = list()
    with open(filename) as file:
        for line in file:
            line = line.strip()
            elements = line.split(split_char)
            # print(elements)
            elements[0] = int(elements[0]) / 10000000     # Time
            elements[2] = chr(int(elements[2], 16))       # ascii (hex)
            elements[3] = int(elements[3], 16)            # scan code
            # print("elements:", elements)
            lst.append(elements)
    return lst


def txt_reader(file_addr):
    keystrokes = []
    up_times = []
    down_times = []
    df_list = split_line(file_addr)
    for i in range(int((len(df_list)) / 2)):
        keystrokes.append(df_list[2 * i][2])
        up_times.append(df_list[2 * i + 1][0])
        down_times.append(df_list[2 * i][0])
    return up_times, down_times


if __name__ == '__main__':
    # folder = r"E:\alireza\augusta\codes\recorder-english\dataset\638238337645941584467/"
    folder = r"E:\alireza\augusta\codes\recorder-english\dataset\new/"
    up_times, down_time = txt_reader(folder + r"words.txt")
    n_seconds_of_silence = 6
    remove_from_start_time = 1
    time, energy, s_time, signal, sound_duration = energy_analyser(folder + r"words.wav",
                                                                   n_seconds_of_silence, remove_from_start_time)
    segments, best_threshold = find_best_threshold(energy, time, best_threshold=0.5, factor=0.1, num_desired_keystrokes=205)
    inter_keystroke_intervals, all_max = key_detection(segments)

    plt.figure(figsize=(12, 4))
    # plt.subplot(2, 1, 1)
    plt.plot(s_time, signal, color='b', label='Sound Signal')
    plt.ylabel('Amplitude')
    plt.ylim([min(signal), max(signal)])
    plt.legend()
    # plt.subplot(2, 1, 2)
    plt.plot(time, energy, color='r', label='Energy (RMSE)')
    # plt.ylim([-0.1, 0.1])
    plt.xlabel('Time (s)')
    # plt.ylabel('Energy')
    plt.legend()
    s_mean = mean(signal)
    s_max = max(signal)
    s_min = abs(min(signal))
    print(s_max, s_mean, s_min)
    print(up_times)
    print(down_time)
    print(segments)
    print(len(segments))
    print(best_threshold)
    x = np.ones(len(time))
    print(x)
    plt.plot( time, best_threshold * x, color='g', label='threshold')
    for i in range(len(up_times)):
        plt.axvspan(down_time[i] - n_seconds_of_silence, up_times[i] - n_seconds_of_silence,  ec='yellow', color='yellow')
    for i in range(len(all_max)):
        plt.axvspan(time[all_max[i]], time[all_max[i]], ec='red', color='green')

    plt.show()

    for i in range(len(up_times)-1):
        print("number", i)
        print("diff up times:", up_times[i+1] - up_times[i])
        print("diff down times:", down_time[i+1] - down_time[i])
        print("diff max (interval)", time[all_max[i+1]] - time[all_max[i]])
        delta = ((down_time[i + 1] + up_times[i + 1]) / 2) - ((down_time[i] + up_times[i]) / 2)
        print("delta for train:", delta)
