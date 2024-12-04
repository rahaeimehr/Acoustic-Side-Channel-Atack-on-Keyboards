import os
import pandas as pd
import librosa
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set data preproccesing args', add_help=False)
    parser.add_argument('--n_fft', type=int, default=2048, help='n_fft: number of samples used for fft')
    parser.add_argument('--hop_length', type=int, default=512, help='hop_length: number of samples between successive frames')
    parser.add_argument('--n_mels', type=int, default=64, help='number of mel bands')

    parser.add_argument('--resampling_mode', type=str, default='resample minority', 
                        help='resampling mode: resmaple minority, down sampel majority, data augmentation',
                        choices=['resample minority', 'down sampel majority', 'data augmentation'])
    parser.add_argument('--resampling_order', type=str, default='before', 
                        help='resampling order: before or after spliting data', 
                        choices=['before', 'after'])

    parser.add_argument('--data_path', type=str, required=True, help='path to data')
    parser.add_argument('--output_path', type=str, required=True, help='path to save preproccesed data')

    return parser

def label_audio(times_ms, timestamps, window_size=50):
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
    press_times = timestamps['down'].values  # Assuming column is 'time_in_ms'

    window_size = times_ms[1] - times_ms[0]
    
    for press_time in press_times:
        # Find frames within the window around the key press time
        window_start = press_time - window_size / 2
        window_end = press_time + window_size / 2
        active_frames = np.where((times_ms >= window_start) & (times_ms <= window_end))[0]
        
        # Label those frames as 1 (key pressed)
        labels[active_frames] = 1
    
    return labels

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
    
    # Convert frames to time in milliseconds
    times = librosa.frames_to_time(np.arange(mel_spectrogram_db.shape[1]), sr=sr, hop_length=hop_length)
    # times_ms = times * 1000  # Convert to milliseconds
    times_ms = times
    
    return mel_spectrogram_db, times_ms

# Extract features for each audio file

def read_data(data_path):
    data_dir = data_path
    # List to store all loaded data
    audio_data = []
    press_times = []

    # Loop through all files in the directory
    for filename in os.listdir(data_dir):
        # Check if the file is a .wav file
        if filename.endswith('.wav'):
            # Load the audio file
            filepath = os.path.join(data_dir, filename)
            audio, sr = librosa.load(filepath, sr=None)  # Load with original sampling rate
            
            # Extract the word index to find the corresponding .xlsx file
            word_index = filename.split('_')[1].split('.')[0]
            if word_index == '0':
                continue
            
            xlsx_filename = f'word_{word_index}.xlsx'
            
            # Load the corresponding Excel file
            xlsx_filepath = os.path.join(data_dir, xlsx_filename)
            timestamps = pd.read_excel(xlsx_filepath)
            
            # Store audio data and timestamps
            audio_data.append((audio, sr))
            press_times.append(timestamps)   

    return {'audio_data': audio_data, 'press_times': press_times, 'timestamps': timestamps}

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

def normalize_data(X_train, X_test):
    # Normalize features (zero mean, unit variance)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Normalize the training and test data
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std

    return X_train_normalized, X_test_normalized, mean, std

def preproceed_data(data_path , output_path, n_fft, n_mels, hop_length, resampling_order, save = False, verbose = 0):

    if verbose > 0:
        print("Preprocessing data...")
        if verbose > 1:
            print("Data path:", data_path)
            print("Output path:", output_path)
            print("n_fft:", n_fft)
            print("n_mels:", n_mels)
            print("hop_length:", hop_length)
            print("resampling_order:", resampling_order)


    # read data from data_path
    data = read_data(data_path)
    audio_data = data['audio_data']
    press_times = data['press_times']

    # Extract features for each audio file
    features = []
    times_list = []
    for (audio, sr), timestamps in zip(audio_data, press_times):
        mel_features, times_ms = extract_features(audio, sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
        features.append(mel_features)
        times_list.append(times_ms)

    # labling data
    labels = []
    for times_ms, timestamps in zip(times_list, press_times):
        labels.append(label_audio(times_ms, timestamps))

    
    # Prepare the frame-wise data
    X_train, y_train = prepare_framewise_data(features, labels)
    raw_X_train = X_train.copy()  # Keep a copy of the raw features for visualization
    raw_y_train = y_train.copy()  # Keep a copy of the raw labels for visualization

    # split before resampling or after that 
    if resampling_order == 'before':
        X_train, y_train = resmapeling_minority(X_train, y_train)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    elif resampling_order == 'after':
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, y_train = resmapeling_minority(X_train, y_train)
        X_test, y_test = resmapeling_minority(X_test, y_test)

    # Normalize the data
    X_train_normalized, X_test_normalized, mean, std = normalize_data(X_train, X_test)
    
    # Convert the balanced data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


    if save:
        # Save the data to disk
        output_dir = output_path + '/model_data' # ../data/pipeline/model_data
        np.save(os.path.join(output_dir, 'X_train.npy'), raw_X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), raw_y_train)

        # Alternative: Saving as pickle
        with open(os.path.join(output_dir, 'training_data.pkl'), 'wb') as f:
            pickle.dump((raw_X_train, raw_y_train), f)

        
        output_path = output_path + '/tensors' # ../data/pipeline/tensors
        torch.save((X_train_tensor, y_train_tensor), output_path + 'train_data.pt')
        torch.save((X_test_tensor, y_test_tensor), output_path + 'test_data.pt')

        # Optionally, save the normalization parameters (mean and std)
        torch.save((mean, std), output_path + 'normalization_params.pt')

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
 
        
def main(args):
    preproceed_data(data_path=args.data_path, 
                    output_path=args.output_path, 
                    n_fft=args.n_fft, 
                    n_mels=args.n_mels, 
                    hop_length=args.hop_length , 
                    resampling_order=args.resampling_order,
                    save=True, verbose=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('preproccesing script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)