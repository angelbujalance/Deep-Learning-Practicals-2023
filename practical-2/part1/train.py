################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models

    model = models.resnet18(weights='IMAGENET1K_V1')

    # Randomly initialize and modify the model's last layer for CIFAR100.
    for param in model.parameters():
        param.requires_grad = False # Parameters will not be updated during training
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Replaces last FC layer
    nn.init.normal_(model.fc.weight.data, 0, 0.01)
    nn.init.zeros_(model.fc.bias.data)
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir=data_dir, augmentation_name=augmentation_name)

    # Adjust the batch size of the train_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(params=model.fc.parameters(),lr=lr)
    loss_module = nn.CrossEntropyLoss() # Loss module or criterion

    # Lists to store relevant statistics for training and validation
    training_loss = []
    val_accuracies = []

    # Accuracy of the best model during accuracy. Set to 0 before training
    best_val_acc = 0.0

    print("The training loop starts")
    # Training loop with validation after each epoch. Save the best model.
    for epoch in range(epochs):

        # Training loop
        model.train()

        train_loss = 0.0
        for step, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_module(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if step % 200 == 199: # Append every 200 mini-batches
                training_loss.append(train_loss)
                train_loss = 0.0

        # Validation loop
        model.eval()

        val_loss = 0.0
        val_acc = 0
        count = 0
        for (inputs, labels) in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)

                loss = loss_module(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
                count += labels.shape[0]
        val_acc /= count
        val_accuracies.append(val_acc)

        # Saving the best model during validation
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_name)

    model.load_state_dict(torch.load(checkpoint_name))
    print("Validation accuracies for epoch:", val_accuracies)
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    #loss_module = nn.CrossEntropyLoss() # Loss module or criterion

    # Metrics for validation
    accuracy = 0
    count = 0

    # Validation loop. Loops over the dataset and computes the accuracy
    for (inputs, labels) in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

            accuracy += (outputs.argmax(1) == labels).sum().item()
            count += labels.shape[0]
    accuracy /= count

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()
    model.to(device)

    # Get the augmentation to use
    if augmentation_name is not None:
        model_name = f'ResNet18 with augmentation method {augmentation_name}'
        checkpoint_name = f'../part2/models/best_model_validation_resnet18_{augmentation_name}.pt'
    else:
        model_name = f'ResNet18 without augmentation methods'
        checkpoint_name = f'../part2/models/best_model_validation_resnet18_default.pt'

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name)

    # Evaluate the model on the test set
    cifar100_test = get_test_set(data_dir, test_noise)
    test_loader = data.DataLoader(dataset=cifar100_test, batch_size=batch_size, shuffle=False,
                                  drop_last=False)
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'The accuracy of the {model_name} on test set is {test_accuracy: .2f}')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
