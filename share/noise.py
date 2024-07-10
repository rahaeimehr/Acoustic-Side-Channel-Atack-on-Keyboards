import pyaudio
import wave
import noisereduce as nr
from scipy.io import wavfile as wav
import numpy as np

# define the audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# create an instance of the PyAudio class
audio = pyaudio.PyAudio()

# start recording
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording started...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# stop recording
stream.stop_stream()
stream.close()
audio.terminate()

# save the recording to a WAV file
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

# load the WAV file into an audio clip
# with wave.open(WAVE_OUTPUT_FILENAME, 'rb') as wave_file:
#     audio_clip = wave_file.readframes(-1)

sampling_rate, sound_data = wav.read(WAVE_OUTPUT_FILENAME)


if type(sound_data[0]) == np.ndarray:
    sound_data = sound_data[:, 0]


# reduce the noise in the audio clip

reduced_noise = nr.reduce_noise(sound_data, RATE)

# save the noise-reduced audio clip to a new WAV file
REDUCED_NOISE_OUTPUT_FILENAME = "reduced_noise_output.wav"
with wave.open(REDUCED_NOISE_OUTPUT_FILENAME, 'wb') as wave_file:
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(reduced_noise)
    
print("Noise reduction complete. Output saved to", REDUCED_NOISE_OUTPUT_FILENAME)
