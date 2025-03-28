import sys
import os
import torch
import argparse

import matplotlib.pyplot as plt
import numpy as np

from raw_data_preproccesing import label_audio, extract_features, prepare_framewise_data, normalize_data, read_raw_data

import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))  # Output probability between 0 and 1
        return x


def keep_first_every_four(tensor, gap=4):
        result = torch.zeros_like(tensor)
        i = 0
        n = tensor.size(0)

        while i < n:
            if tensor[i] == 1:
                result[i] = 1
                i += 1
                count = 1
                while i < n and tensor[i] == 1:
                    if count % gap == 0:
                        result[i] = 1
                    i += 1
                    count += 1
            else:
                i += 1
        return result

def get_args_parser():
    parser = argparse.ArgumentParser('Set data preprocessing args', add_help=False)
    
    # Add arguments for preprocessing
    parser.add_argument('--n_fft', type=int, default=2048, help='n_fft: number of samples used for FFT')
    parser.add_argument('--hop_length', type=int, default=512, help='hop_length: number of samples between successive frames')
    parser.add_argument('--n_mels', type=int, default=64, help='number of Mel bands')

    parser.add_argument('--path', type=str, required=True, 
                        help='the path containing audio and timestamps (space-separated)')
    parser.add_argument('--audio_number', type=int, default=0, 
                        help='the path containing audio and timestamps (space-separated)')
    parser.add_argument('--model_path', type=str, required=True,help='Path to the model checkpoint')
    parser.add_argument('--show_plot', action='store_true', help='Show plot of audio and predictions')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for classification')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    
    return parser

def inference(args):
    
    path = args.path
    # read data from data_folders
    # data = read_data_multiple_folders(data_folders)
    data = read_raw_data(path)
    audio_data = data['audio_data']
    press_times = data['press_times']

    # Extract features for each audio file
    print('processing audio files...')
    n_fft = args.n_fft
    n_mels = args.n_mels
    hop_length = args.hop_length
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

    X, y = [], []
        
    for mel_features, label in zip(features, labels):
            # Flatten along time axis
            X.append(mel_features.T)  # Transpose so each row is a time frame (n_mels,)
            y.append(label)

    X_train = X
    y_train = y

    print('loading model...')
    model_path = args.model_path

    checkpoint_model = torch.load(model_path)

    model_state_dict = checkpoint_model['model_state_dict']
    input_size = checkpoint_model['input_size']

    # read the file 
    mean = checkpoint_model['mean']
    std = checkpoint_model['std']

    norm_X_train = []
    for x in X_train:
        norm_X_train.append((x - mean) / std)

    model = MyModel(input_size)

    # Load the state dictionary
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode
    model.eval()

    print('inference...')
    exp_number = args.audio_number
    threshold = args.threshold
    input_data = torch.tensor(norm_X_train[exp_number], dtype=torch.float32)

    predict_ligit = model(input_data)
    predictions = (predict_ligit >= threshold).int().squeeze()
    true_labels = torch.tensor(y_train[exp_number], dtype=torch.int)

    accuracy = (predictions == true_labels).float().mean()

    # Correct Calculation of True Positives
    true_positive = ((predictions == 1) & (true_labels == 1)).sum().item()

    # Count the number of predicted positives
    predicted_positive = (predictions == 1).sum().item()

    # Avoid division by zero for precision calculation
    precision = true_positive / (predicted_positive + 1e-8) if predicted_positive > 0 else 0.0

    print(f"Accuracy: {accuracy.item():.4f}")
    print(f"Precision: {precision:.4f}")

    post_prediction = keep_first_every_four(predictions)
    print(post_prediction)
    # tensor([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1])
    if args.show_plot:
         # Create time axes

        plot_audio = audio_data[exp_number][0]
        audio_time = np.linspace(0, len(plot_audio), len(plot_audio))
        prediction_time = np.linspace(0, len(plot_audio), len(predictions))  # Scale to match audio timeline

        # Plot audio
        plt.figure(figsize=(12, 4))
        plt.plot(audio_time, plot_audio, color='black', label="Audio Signal")

        for i, pred in enumerate(predictions):
            if pred == 1:
                plt.axvspan(prediction_time[i], prediction_time[i + 1] if i + 1 < len(predictions) else prediction_time[i], 
                            color='red', alpha=0.3, label="Predicted Label" if i == 0 else "")

        # Overlay horizontal bars for true labels (Blue)
        for i, true in enumerate(true_labels):
            if true == 1:
                plt.axvspan(prediction_time[i], prediction_time[i + 1] if i + 1 < len(true_labels) else prediction_time[i], 
                            color='blue', alpha=0.3, label="True Label" if i == 0 else "")

        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Audio Signal with Prediction and True Label Highlights")

        plt.show()

    mask = keep_first_every_four(post_prediction)

    # Use the mask to filter tensor2
    filtered_values = times_list[3][mask == 1]

    # save the filtered values to a file
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'predicted_timesteps.txt')
    with open(output_file, 'w') as f:
        for value in filtered_values:
            f.write(f"{value}\n")
    print(f"Filtered values saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    inference(args)