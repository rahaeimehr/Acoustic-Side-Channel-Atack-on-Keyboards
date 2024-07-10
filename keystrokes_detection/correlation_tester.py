import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import correlation

output_dir = r'E:\alireza\augusta\codes\net5\dataset\638126912302862500573/keystrokes/'

# load two audio signals
signal1, sample_rate1 = librosa.load(output_dir + 'keystroke_1.wav', sr=None)
signal2, sample_rate2 = librosa.load(output_dir + 'keystroke_2.wav', sr=None)

# Calculate the correlation-based distance
corr_dist = (np.correlate(signal1, signal2, mode='full'))

# Calculate the correlation coefficient
r = np.corrcoef(signal1, signal2)[0, 1]
# Calculate the distance
distance = 1 - r
print(distance)
# Plot the result
plt.plot(corr_dist)
plt.xlabel('Sample')
plt.ylabel('Correlation-based distance')
plt.show()