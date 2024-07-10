"""
Written by Alireza Taheri ( March 2022 )

"""

from datetime import datetime

import pyaudio
import wave
import keyboard
from database_manager import DbHandler


def record(file_name):
    # defining audio variables
    today = datetime.today()
    sound_address = f"{file_name}/main.wav"
    # print(f"Sound File Path: {sound_address}")
    db_file = f"{file_name}/main.db"
    db = DbHandler(db_file)

    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    Y = 100

    # Calling pyadio module and starting recording
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=chunk)

    stream.start_stream()
    # time_start = datetime.timestamp(datetime.now())
    # print("Sound Record is Starting!")
    print("Please DO NOT TYPE, the recorder is sampling your environmental noise !!!")

    # Running Keylogger in a separate
    # Recording data until under threshold
    frames = []
    start = True
    start2 = True

    while True:
        if keyboard.is_pressed("ESC"):
            print("You pressed ESC")
            break
        # Converting chunk data into integers
        if start:
            import time
            s = time.time()
            time_start = datetime.timestamp(datetime.now())
            start = False

        if start2 and (time.time() - s >= 7):
            print("Now you can type...")
            start2 = False

        data = stream.read(chunk)
        # data_int = struct.unpack(str(2 * chunk) + 'B', data)
        # Finding average intensity per chunk
        # avg_data = sum(data_int) / len(data_int)
        # print(str(avg_data))
        # Recording chunk data
        frames.append(data)

        # if avg_data < Y:
        #     break

    # Stopping recording
    time_end = datetime.timestamp(datetime.now())
    stream.stop_stream()
    stream.close()
    p.terminate()
    # print("Ending recording!")
    # Saving file with wave module
    wf = wave.open(sound_address, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    db.db_insert_start_end([(time_start, time_end)])
