#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
# train_model.py

# TODO: Import your dependencies.


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from sagemaker.debugger import DebuggerHookConfig
from smdebug import modes
import torch.profiler
import os
from torch.utils.data import DataLoader
import logging
import sys

import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss criterion (CrossEntropy for classification tasks)
criterion = nn.CrossEntropyLoss()

def test(model, test_loader,device,criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''    
    hook.set_mode(smd.modes.EVAL)

    model.to(device)
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = running_loss / total
    accuracy = correct / total

    logger.info(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    return avg_loss, accuracy

def train(model, train_loader, criterion, optimizer, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.to(device)
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    hook.set_mode(smd.modes.EVAL)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Debugging hook
        hook.save_tensor("train_loss", loss)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    logger.info(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')

    return avg_loss, accuracy
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer for dog breed classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)  # 133 dog breeds in the dataset

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    A DataLoader is a PyTorch utility essential for efficiently managing large
    datasets that exceed memory capacity, enabling batched input for deep
    learning models, and supporting features like on-the-fly shuffling and
    parallel data loading to accelerate training. It automates critical
    preprocessing steps—such as batch creation, optional epoch-wise shuffling,
    and multiprocess data fetching—while optimizing throughput by reducing GPU
    idle time, making it indispensable for scalable and performant model 
    training.
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(data, 'train'), transform=transform)
    valid_data = datasets.ImageFolder(os.path.join(data, 'valid'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data, 'test'), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def main(args):

    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Training the model.")

    model=train(model, train_loader, loss_criterion, optimizer,hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model.")

    test(model, test_loader, criterion,hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")

    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':

    '''
    TODO: Specify any training args that you might need
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data-dir', type=str, default='s3://sagemaker-us-east-1-593793021221/dogImages/', help='S3 path containing train/valid/test folders, e.g., s3://sagemaker-us-east-1-93793021221/dogImages/')
    parser.add_argument('--model-dir', type=str, default='s3://sagemaker-us-east-1-593793021221/ComputerVision/Model/', help='S3 path to save the trained model')

    args = parser.parse_args()
    main(args)
