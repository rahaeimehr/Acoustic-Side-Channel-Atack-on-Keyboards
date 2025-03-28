import os
import pandas as pd
import librosa
import numpy as np
import pickle
import copy

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import DataLoader, TensorDataset

import argparse
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class MLPmodelSigmoidhead(nn.Module):
    def __init__(self, input_size):
        super(MLPmodelSigmoidhead, self).__init__()
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

def get_args_parser():
    parser = argparse.ArgumentParser('Set data model training args', add_help=False)

    parser.add_argument('--data_path', type=str, required=True, help='path to data')
    parser.add_argument('--output_path', type=str, required=True, help='path to save preproccesed data')

    parser.add_argument('--save', action='store_true', help='save the model')
    
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser

def load_data(data_path):
    # Load the preprocessed and saved data
    X_train_tensor, y_train_tensor = torch.load(os.path.join(data_path, 'train_data.pt'))
    X_test_tensor, y_test_tensor = torch.load(os.path.join(data_path, 'test_data.pt'))

    # Load normalization parameters (mean, std)
    mean, std = torch.load(os.path.join(data_path, 'normalization_params.pt'))

    # You can also check the loaded data if you want
    print(f"Loaded training data: X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")
    print(f"Loaded test data: X_test shape: {X_test_tensor.shape}, y_test shape: {y_test_tensor.shape}")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, mean, std

def get_class_weights(labels):
    labels_np = labels.cpu().numpy().flatten()
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels_np)
    return torch.tensor(class_weights, dtype=torch.float32)

def train(data_path, output_path, save=False, save_fig=False, verbose=2, device='cuda'):
    # Load the preprocessed and saved data
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, mean, std = load_data(data_path)

    # Initialize the model, loss function, and optimizer
    input_size = X_train_tensor.shape[1]
    model = MLPmodelSigmoidhead(input_size=input_size)
    class_weights = get_class_weights(y_train_tensor).to('cpu')
    print(f"Class weights: {class_weights}")
    
    # Define the loss function with reduction='none'
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    precisions = []

    from tqdm import tqdm

    best_precision = 0.0  # Initialize the best precision
    best_test_loss = 100  # Initialize the best test loss
    best_model_state = None  # To store the best model's state

    num_epochs = 20
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap the train_loader with tqdm to display progress
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss_unweighted = criterion(outputs, y_batch)
            
            # Compute batch-specific weights:
            # Weight = class_weights[1] for positive samples and class_weights[0] for negative samples
            batch_weights = class_weights[1] * y_batch + class_weights[0] * (1 - y_batch)
            
            # Compute weighted loss and average it
            loss = (loss_unweighted * batch_weights).mean()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        true_positives = 0
        predicted_positives = 0

        with torch.no_grad():
            # Wrap the test_loader with tqdm for a progress bar during evaluation
            for X_batch, y_batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing", leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                loss_unweighted = criterion(outputs, y_batch)
                batch_weights = class_weights[1] * y_batch + class_weights[0] * (1 - y_batch)
                loss = (loss_unweighted * batch_weights).mean()
                test_loss += loss.item() * X_batch.size(0)
                
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                true_positives += ((predicted == 1) & (y_batch == 1)).sum().item()
                predicted_positives += (predicted == 1).sum().item()
            
            test_loss = test_loss / len(test_loader.dataset)
            test_accuracy = correct / total
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            precisions.append(precision)

            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test precision: {precision:.4f}")

            # Check if current epoch achieved a new best precision
        if precision > best_precision:
            best_precision = precision
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model found at epoch {epoch+1} with precision: {precision:.4f}")


    if save:
        model_output = os.path.join(output_path, 'models/')
        # Save the model and related parameters
        torch.save({
            'model_state_dict': best_model_state,
            'input_size': input_size,
            'mean': mean,
            'std': std,
        }, os.path.join(output_path, 'model_checkpoint.pth'))
        print("Model saved successfully.")

    if save_fig:
        import matplotlib.pyplot as plt
        # Plot training and test losses
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Test loss')
        plt.legend()
        fig_path = os.path.join('..', 'figs')
        plt.savefig(os.path.join(fig_path, 'loss_plot.png'))
        plt.clf()

        # Plot training and test accuracies
        plt.plot(train_accuracies, label='Training accuracy')
        plt.plot(test_accuracies, label='Test accuracy')
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'accuracy_plot.png'))
        plt.clf()

def main(args):
    train(data_path=args.data_path, 
          output_path=args.output_path,
          save=args.save , save_fig=True, verbose=2, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('model training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
