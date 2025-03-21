import queue
import sys
import torch
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# Load Whisper model (use "tiny", "small", "medium", or "large")
model_size = "small"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio_queue = queue.Queue()

def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return None, pyaudio.paContinue

# Start recording
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=CHUNK, stream_callback=callback)
stream.start_stream()

print("Listening... Speak now!")

try:
    while True:
        if not audio_queue.empty():
            audio_data = np.frombuffer(audio_queue.get(), np.int16).astype(np.float32) / 32768.0
            segments, _ = model.transcribe(audio_data, beam_size=5)
            for segment in segments:
                print(segment.text)

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit()
