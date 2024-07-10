import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load audio file
filename = r'E:\alireza\augusta\codes\net5\dataset\638126912302862500573/words.wav'
sampling_rate, audio_data = wavfile.read(filename)

# Define window size and step size
window_size = 440
step_size = 220

# Create Hamming window function
hamming_window = np.hamming(window_size)
# plt.plot(hamming_window)
# plt.show()

# Initialize array for storing FFT coefficients sum
fft_sum = np.zeros(len(audio_data) // step_size)

# Loop through audio data and calculate FFT coefficients sum
for i in range(0, len(audio_data) - window_size, step_size):
    # Apply Hamming window to audio data
    windowed_data = audio_data[i:i + window_size] * hamming_window

    # Calculate FFT coefficients and add to sum
    fft_coefficients = np.fft.rfft(windowed_data)
    fft_sum[i // step_size] = np.sum(np.abs(fft_coefficients[4:100]))

# Define frequency range of interest
freq_range = np.arange(0, sampling_rate * 4, sampling_rate / window_size)[:len(fft_sum)]

# Plot FFT coefficients sum
plt.plot( fft_sum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sum of FFT Coefficients')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sample rate and duration
sr = 44100
duration = 2

# Generate a time array in seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Generate a signal with two frequencies
signal = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t)

# Define the window size and padding
window_size = 440
padding = window_size - len(signal) % window_size

# Pad the signal with zeros if necessary
if padding > 0:
    signal = np.append(signal, np.zeros(padding))

# Reshape the signal into overlapping windows
windows = signal.reshape(-1, window_size)

# Apply a Hamming window to each window
hamming_window = np.hamming(window_size)
windows *= hamming_window

# Calculate the FFT of each window
fft = np.fft.rfft(windows)

# Calculate the power spectrum
power = np.abs(fft) ** 2

# Calculate the frequency axis
freq = np.fft.rfftfreq(window_size, 1/sr)

# Sum up the power spectrum within the specified frequency range
start_freq = 400
end_freq = 22000
start_bin = np.argmin(np.abs(freq - start_freq))
end_bin = np.argmin(np.abs(freq - end_freq))
sum_power = np.sum(power[:, start_bin:end_bin], axis=1)

# Plot the results
plt.plot(t[:len(sum_power)], sum_power)
plt.xlabel('Time (s)')
plt.ylabel('Sum of FFT coefficients')
plt.show()
