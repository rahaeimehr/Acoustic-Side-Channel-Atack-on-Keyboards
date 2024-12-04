import os
import pandas as pd
import librosa
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    parser.add_argument('--output_path', type=str, required=False, help='path to save preproccesed data')

    return parser

def load_data(data_path):
    # Load the preprocessed and saved data
    X_train_tensor, y_train_tensor = torch.load(data_path + 'train_data.pt')
    X_test_tensor, y_test_tensor = torch.load(data_path + 'test_data.pt')

    # Load normalization parameters (mean, std)
    mean, std = torch.load(data_path + 'normalization_params.pt')

    # You can also check the loaded data if you want
    print(f"Loaded training data: X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")
    print(f"Loaded test data: X_test shape: {X_test_tensor.shape}, y_test shape: {y_test_tensor.shape}")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, mean, std

def train(data_path, output_path, save=False,save_fig=False, verbose=2):
    # Load the preprocessed and saved data
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, mean, std = load_data(data_path)

    # Initialize the model, loss function, and optimizer
    input_size = X_train_tensor.shape[1]
    model = MLPmodelSigmoidhead(input_size=input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    criterion.to(device)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        # train model
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Zero gradients

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)

            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"Train accuracy: {correct / total:.4f}")
        train_losses.append(epoch_loss)
        train_accuracies.append(correct / total)

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
                
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            test_loss = test_loss / len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accuracies.append(correct / total)

    if save:
        model_output = output_path + 'models/'
        # Save the model after training if needed
        # Save the model and other useful info
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'mean': mean,  # Optional: save normalization parameters
            'std': std,
        }, output_path + 'model_checkpoint.pth')
        print("Model saved successfully.")

    if save_fig:
        # plot the training and test losses
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Test loss')
        plt.legend()
        
        # Save the plot
        fig_path = '../figs/'
        plt.savefig(fig_path + 'loss_plot.png')
        # clear the current figure
        plt.clf()

        #plot the training and test accuracies
        plt.plot(train_accuracies, label='Training accuracy')
        plt.plot(test_accuracies, label='Test accuracy')
        plt.legend()

        # Save the plot
        fig_path = '../figs/'
        plt.savefig(fig_path + 'accuracy_plot.png')
        # clear the current figure
        plt.clf()


def main(args):
    train(data_path=args.data_path, 
          output_path=args.output_path,
          save=False, save_fig=True, verbose=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('model training script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)