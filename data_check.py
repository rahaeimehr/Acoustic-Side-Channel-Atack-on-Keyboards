import numpy as np
import matplotlib.pyplot as plt
import wave


def get_keys_down_up(file_path):
    lst = list()
    turn = 0
    split_char = ','
    key = [0, 0, '']
    i = 0
    with open(file_path+".txt") as file:            
        for line in file:
            i += 1
            line = line.strip()
            elements = line.split(split_char)
            if int(elements[1]) == turn:   
                key[turn] = int(elements[0]) / 10000000                  
                key[2] = chr(int(elements[2], 16))
                if turn == 1:
                    lst.append(key)
                    key = [0, 0, '']
                turn ^= 1
            else:
                print("Error in line:", i, line) 
    return lst
    

def analyze_recordings(file_path):
    try:
        # Open the WAV file
        wav_file_path = file_path+'.wav'
        with wave.open(wav_file_path, 'rb') as wav_file:
            # Retrieve file properties
            n_channels = wav_file.getnchannels()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()  # Get sample width in bytes

            # Read raw audio data
            raw_audio = wav_file.readframes(n_frames)

        # Initialize audio_data to prevent errors
        audio_data = None  

        # Choose correct NumPy dtype
        if sample_width == 1:
            dtype = np.uint8  # 8-bit PCM (unsigned)
            audio_data = np.frombuffer(raw_audio, dtype=dtype)
        elif sample_width == 2:
            dtype = np.int16  # 16-bit PCM (signed)
            audio_data = np.frombuffer(raw_audio, dtype=dtype)
        elif sample_width == 3:
            # Handle 24-bit PCM separately
            raw_audio = np.frombuffer(raw_audio, dtype=np.uint8)  # Read as bytes
            raw_audio = raw_audio.reshape(-1, 3)  # Reshape to (num_samples, 3)

            # Convert little-endian 24-bit to signed 32-bit int
            audio_data = np.left_shift(raw_audio[:, 2], 16) | np.left_shift(raw_audio[:, 1], 8) | raw_audio[:, 0]
            audio_data = audio_data.astype(np.int32)  # Convert to int32
            
            # Adjust for signed values
            audio_data[audio_data >= (1 << 23)] -= (1 << 24)
        elif sample_width == 4:
            dtype = np.int32  # 32-bit PCM
            audio_data = np.frombuffer(raw_audio, dtype=dtype)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        if audio_data is None:
            raise RuntimeError("Failed to load audio data properly.")

        print(f"Audio Data Shape: {audio_data.shape}")

        # If the audio is stereo (2 channels), select only one channel
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2)  # Convert into (samples, channels)
            audio_data = audio_data[:, 0]  # Select only the first channel

        # Create the time axis in seconds
        time = np.linspace(0, len(audio_data) / frame_rate, num=len(audio_data))

        lst = get_keys_down_up(file_path)

        # Plot the waveform
        plt.figure(figsize=(10, 5))
        plt.plot(time, audio_data, color='b')
        plt.title('Audio Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        offset = 0.045
        for key in lst:
            if key[2] == chr(0):
                plt.axvspan(key[0] + offset, key[1]+offset, color='red')
            else:
                plt.axvspan(key[0]+offset, key[1]+offset, color='darkgray')          
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please provide a valid file name or path.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    analyze_recordings('dataset2/638754206430228762355/words')
