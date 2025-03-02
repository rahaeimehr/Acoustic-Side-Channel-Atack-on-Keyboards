import os
import pandas as pd
import librosa
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import argparse

# seed=42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
def get_args_parser():
    parser = argparse.ArgumentParser('Set data preprocessing args', add_help=False)
    
    # Add arguments for preprocessing
    parser.add_argument('--n_fft', type=int, default=2048, help='n_fft: number of samples used for FFT')
    parser.add_argument('--hop_length', type=int, default=512, help='hop_length: number of samples between successive frames')
    parser.add_argument('--n_mels', type=int, default=64, help='number of Mel bands')
    
    parser.add_argument('--data_folders', type=str, nargs='+', required=False, 
                        help='List of folder paths containing audio and timestamps (space-separated)')
    
    parser.add_argument('--path', type=str, required=True, 
                        help='the path containing audio and timestamps (space-separated)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    parser.add_argument('--resampling_type', type=str, choices=['majority', 'minority', 'None'], default='None', 
                        help='type of resampling for balancing data')
    parser.add_argument('--resampling_order', type=str, choices=['before', 'after'], default='after', 
                        help='Order of resampling (before or after train-test split)')
    parser.add_argument('--save', action='store_true', help='Save the preprocessed data')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose level (0 = silent, 1 = basic, 2 = detailed)')
    
    return parser

def label_audio(times_ms, timestamps , overlap, shift = 0.02 , window_size=50 ):
    """
    Labels each time frame in the audio based on key press timestamps.
    
    Parameters:
    - times_ms: Times of each frame in milliseconds
    - timestamps: DataFrame with key press times
    - window_size: Time window in milliseconds to mark a key press as "active"
    
    Returns:
    - labels: Binary labels (1 for key press, 0 for no key press) for each frame
    """
    labels = np.zeros(len(times_ms))
    timestamps = timestamps[timestamps['event'] == 0].copy()

    # Divide the timestamp by 1e7
    timestamps['time_step'] = timestamps['time_step'] / 1e7 + shift
    press_times = timestamps['time_step'].values

    window_size = times_ms[1] - times_ms[0]

    for press_time in press_times:
        # Find frames within the window around the key press time
        window_start = press_time 
        window_end = press_time + window_size * overlap
        active_frames = np.where((times_ms >= window_start) & (times_ms <= window_end))[0]

        labels[active_frames] = 1
    
    return labels

# Extract features for each audio file
def extract_features(audio, sr, n_fft=2048, n_mels=64, hop_length=512):
    """
    Extracts Mel spectrogram features from audio.
    
    Parameters:
    - audio: Audio signal (numpy array)
    - sr: Sampling rate of the audio
    - n_mels: Number of Mel bands
    - hop_length: Number of samples between successive frames
    
    Returns:
    - mel_spectrogram: Mel spectrogram features
    - times_ms: Times of each frame in milliseconds
    """
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Convert frames to time 
    times = librosa.frames_to_time(np.arange(mel_spectrogram_db.shape[1]), sr=sr, hop_length=hop_length)
    times_ms = times

    overlap = int(n_fft / hop_length)
    
    return mel_spectrogram_db, times_ms, overlap


def read_raw_data(path):
    audio_data = []
    press_times = []

    for folder_name in os.listdir(path):
        if folder_name.isdigit():
            folder_path = os.path.join(path, folder_name)
            files = set(os.listdir(folder_path))  # Convert to set for quick lookup
            
            # Find all "text.wav" files in the folder
            for file_name in files:
                if file_name.endswith("text.wav"):
                    text_file = file_name.replace("text.wav", "text.txt")
                    
                    # Ensure both files exist
                    if text_file in files:
                        # Load audio
                        file_path_audio = os.path.join(folder_path, file_name)
                        audio, sr = librosa.load(file_path_audio, sr=None)
                        audio_data.append((audio, sr))
                        
                        # Load text data
                        file_path_text = os.path.join(folder_path, text_file)
                        df_text = pd.read_csv(file_path_text, header=None, usecols=[0, 1], names=['time_step', 'event'])
                        press_times.append(df_text)

                        print(f"reading {file_name} and  {text_file} are in {folder_path}")
                    else:
                        print(f"Skipping {file_name} as {text_file} is missing in {folder_path}")

    return {"audio_data": audio_data, "press_times": press_times}



def prepare_framewise_data(features, labels):
    """
    Flattens each time step of the Mel spectrogram into an independent training sample.
    
    Parameters:
    - features: List of Mel spectrograms (one per audio sample)
    - labels: List of binary label arrays (one per audio sample)
    
    Returns:
    - X: 2D array where each row is a frame (frequency bins) from all audio samples
    - y: 1D array with binary labels for each frame
    """
    X, y = [], []
    
    for mel_features, label in zip(features, labels):
        # Flatten along time axis
        X.append(mel_features.T)  # Transpose so each row is a time frame (n_mels,)
        y.append(label)
    
    # Concatenate all frames and labels into single arrays
    X = np.vstack(X)
    y = np.hstack(y)
    
    return X, y



def resmapeling_minority(X, y):
    # Separate samples by class
    minority_class_indices = np.where(y == 1)[0]
    majority_class_indices = np.where(y == 0)[0]

    # Calculate the imbalance ratio
    num_minority = len(minority_class_indices)
    num_majority = len(majority_class_indices)

    # Number of times we need to repeat the minority samples to match the majority class size
    duplication_factor = num_majority // num_minority
    remaining_samples = num_majority % num_minority  # Handle any remaining samples

    # Oversample the minority class by duplicating the indices
    oversampled_minority_indices = np.hstack([
        np.tile(minority_class_indices, duplication_factor),  # Repeat full sets
        np.random.choice(minority_class_indices, remaining_samples, replace=False)  # Add a few extras
    ])

    # Combine with the majority indices to create a balanced dataset
    balanced_indices = np.hstack([majority_class_indices, oversampled_minority_indices])

    # Shuffle the combined indices to mix classes
    np.random.shuffle(balanced_indices)

    # Use the balanced indices to create balanced X and y
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced

    #downsampling the majority class
def resmapeling_majority(X,y):
    # Separate samples by class
    minority_class_indices = np.where(y == 1)[0]
    majority_class_indices = np.where(y == 0)[0]

    # Calculate the imbalance ratio
    num_minority = len(minority_class_indices)
    num_majority = len(majority_class_indices)

    downsampled_majority_indices=np.random.choice(majority_class_indices,size=num_minority,replace=False)

    # Combine with the minority indices to create a balanced dataset
    balanced_indices = np.hstack([minority_class_indices, downsampled_majority_indices])

    # Shuffle the combined indices to mix classes
    np.random.shuffle(balanced_indices)

    # Use the balanced indices to create balanced X and y
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced



def normalize_data(X_train, X_test):
    # Normalize features (zero mean, unit variance)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Normalize the training and test data
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std

    return X_train_normalized, X_test_normalized, mean, std

# def preproceed_data(data_folders , output_path, n_fft, n_mels, hop_length, resampling_order,resampling_type, save = False, verbose = 0):
def preproceed_data(path , output_path, n_fft, n_mels, hop_length, resampling_order,resampling_type, save = False, verbose = 0):
    """
        Preprocess data from multiple folders.
        
        Parameters:
        - data_folders: List of folder paths containing data (e.g., ["../data/sample1/words", "../data/sample2/words"]).
    
        - Other parameters .
        """
    if verbose > 0:
        print("Preprocessing data...")
        if verbose > 1:
            #print("Data folders:", data_folders)
            print("Path:", path)

            print("Output path:", output_path)
            print("n_fft:", n_fft)
            print("n_mels:", n_mels)
            print("hop_length:", hop_length)
            print("resampling_order:", resampling_order)
            print("resampling_type:", resampling_type)



    # read data from data_folders
    # data = read_data_multiple_folders(data_folders)
    data = read_raw_data(path)
    audio_data = data['audio_data']
    press_times = data['press_times']

    # Extract features for each audio file
    features = []
    times_list = []
    for (audio, sr), timestamps in zip(audio_data, press_times):
        mel_features, times_ms, overlap = extract_features(audio, sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
        features.append(mel_features)
        times_list.append(times_ms)

    # labling data
    labels = []
    for times_ms, timestamps in zip(times_list, press_times):
        labels.append(label_audio(times_ms, timestamps, overlap))

    # Prepare the frame-wise data
    X_train, y_train = prepare_framewise_data(features, labels)
    raw_X_train = X_train.copy()  # Keep a copy of the raw features for visualization
    raw_y_train = y_train.copy()  # Keep a copy of the raw labels for visualization

    # split before resampling or after that 
    if resampling_type =="majority":
        if resampling_order == 'before':
            X_train, y_train = resmapeling_majority(X_train, y_train)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        elif resampling_order == 'after':
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            X_train, y_train = resmapeling_majority(X_train, y_train)
            X_test, y_test = resmapeling_majority(X_test, y_test)

    elif resampling_type == "minority":
        if resampling_order == 'before':
            X_train, y_train = resmapeling_minority(X_train, y_train)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        elif resampling_order == 'after':
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            X_train, y_train = resmapeling_minority(X_train, y_train)
            X_test, y_test = resmapeling_minority(X_test, y_test)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Normalize the data
    X_train_normalized, X_test_normalized, mean, std = normalize_data(X_train, X_test)
    
    # Convert the balanced data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


    if save:
        # Save the data to disk
        output_dir = output_path + '/model_data/' # ../data/pipeline/model_data
        np.save(os.path.join(output_dir, 'X_train.npy'), raw_X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), raw_y_train)

        # Alternative: Saving as pickle
        with open(os.path.join(output_dir, 'training_data.pkl'), 'wb') as f:
            pickle.dump((raw_X_train, raw_y_train), f)

        
        output_path = output_path + '/tensors/' # ../data/pipeline/tensors
        torch.save((X_train_tensor, y_train_tensor), output_path + '/train_data.pt')
        torch.save((X_test_tensor, y_test_tensor), output_path + '/test_data.pt')

        # Optionally, save the normalization parameters (mean and std)
        torch.save((mean, std), output_path + '/normalization_params.pt')

    if verbose > 0:
        print("Data saved successfully.")
        print("X_train shape:", raw_X_train.shape)
        print("y_train shape:", raw_y_train.shape)

        print('saved tensors shapes')
        print("X_train_tensor shape:", X_train_tensor.shape)
        print("y_train_tensor shape:", y_train_tensor.shape)

        print("X_test_tensor shape:", X_test_tensor.shape)
        print("y_test_tensor shape:", y_test_tensor.shape)

        print("acceptable error margin.", times_ms[1] - times_ms[0])
 
    if verbose > 1:
        import matplotlib.pyplot as plt

        # Find indices where label is 1
        ones_indices = np.where(raw_y_train == 1)[0]

        if len(ones_indices) == 0:
            print("No labels with 1 found.")
        else:
            # Randomly pick one index where label is 1
            center_idx = np.random.choice(ones_indices)
            # center_idx = 7075

            # Define slice range around the selected index
            slice_length = 50  # Total slice length
            half_slice = slice_length // 2

            # Ensure the slice stays within bounds
            start_idx = max(0, center_idx - half_slice)
            end_idx = min(raw_X_train.shape[0], center_idx + half_slice)

            # Extract the slice
            mel_slice = raw_X_train[start_idx:end_idx, :]
            labels_slice = raw_y_train[start_idx:end_idx]

            # Plot mel spectrogram
            plt.figure(figsize=(10, 6))
            plt.imshow(mel_slice.T, aspect='auto', origin='lower', cmap='magma')

            # Add vertical lines where labels are 1
            for i, label in enumerate(labels_slice):
                if label == 1:
                    plt.axvline(i, color='cyan', linestyle='--', alpha=0.8)

            # Mark the chosen center label
            center_position = center_idx - start_idx
            plt.axvline(center_position, color='red', linestyle='-', linewidth=2, label="Selected Center Label")

            plt.colorbar(label="Magnitude")
            plt.xlabel("Time Frames")
            plt.ylabel("Frequency Bins")
            plt.title(f"Mel Spectrogram with Labels (Center at {center_idx})")
            plt.legend()
            plt.show()

def main(args):
    preproceed_data(path=args.path, #data_folders=args.data_folders, 
                    output_path=args.output_path, 
                    n_fft=args.n_fft, 
                    n_mels=args.n_mels, 
                    hop_length=args.hop_length , 
                    resampling_type=args.resampling_type,
                    resampling_order=args.resampling_order,
                    save=True, verbose=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('preproccesing script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)