  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      targets: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = 10
    conf_mat = np.zeros((n_classes, n_classes))

    for batch, label in enumerate(targets):
        conf_mat[label][predictions[batch].argmax()] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(conf_mat, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        conf_mat: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = conf_mat.shape[0]
    metrics = {
        'accuracy': np.trace(conf_mat) / np.sum(conf_mat),
        'precision': np.array([conf_mat[n_class, n_class] / np.sum(conf_mat, axis=0)[n_class]
                               for n_class in range(n_classes)]),
        'recall': np.array([conf_mat[n_class, n_class] / np.sum(conf_mat, axis=1)[n_class]
                            for n_class in range(n_classes)])
    }
    metrics['f1_beta'] = ((1 + beta ** 2) *
                          (np.multiply(metrics['precision'], metrics['recall']) /
                           (beta ** 2 * metrics['precision'] + metrics['recall'])))
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    conf_mat = np.zeros(shape=(num_classes, num_classes))

    for (inputs, targets) in data_loader:

        with torch.no_grad():
            # Perform forward pass
            logits = model(inputs)
            batch_conf_mat = confusion_matrix(logits, targets)
            conf_mat += batch_conf_mat

    metrics = confusion_matrix_to_metrics(conf_mat)
    print()
    print(f'Confusion matrix: \n{conf_mat}\n')
    plt.plot()
    plt.matshow(conf_mat)
    for (x, y), value in np.ndenumerate(conf_mat):
        plt.text(x, y, f"{int(value)}", va="center", ha="center")
    plt.savefig("confusion_matrix_pytorch.jpg", dpi=150)

    for beta in [0.1, 1, 10]:
        f1_score = confusion_matrix_to_metrics(conf_mat, beta=beta)['f1_beta']
        print(f'The F-score with beta {beta} is {f1_score}')
        print()

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Initialize model and loss module
    model = MLP(n_inputs=32 * 32 * 3, n_hidden=hidden_dims, n_classes=10, use_batch_norm=use_batch_norm)
    loss_module = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    best_model = None
    best_valid_acc = 0.0

    val_accuracies = []
    log_freq = 10
    logging_info = {'train_loss': [], 'train_acc': [], 'valid_loss': []}

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        count = 0

        # Training
        for step, data in enumerate(cifar10_loader['train'], 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            logits = model(inputs)

            # Calculate training loss & acc
            loss = loss_module(logits, labels)
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            count += labels.shape[0]

            # Perform backprop & SGD update step
            loss.backward()
            optimizer.step()

            if step % log_freq == log_freq - 1:
                logging_info['train_loss'].append(round(train_loss / log_freq, 3))
                logging_info['train_acc'].append(round(train_correct / count, 3))
                train_loss = 0.0

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_acc = 0
        count = 0
        for (inputs, labels) in cifar10_loader['validation']:
            with torch.no_grad():
                # Forward pass
                logits = model(inputs)

                # Metrics
                loss = loss_module(logits, labels)
                valid_loss += loss.item()
                valid_acc += (logits.argmax(1) == labels).sum().item()
                count += labels.shape[0]

        valid_acc /= count
        val_accuracies.append(valid_acc)
        logging_info['valid_loss'].append(round(valid_loss / len(cifar10['validation']), 3))

        # Saving the best model during validation
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)

    # Test
    metrics = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = metrics['accuracy']
    print(f'Test accuracy is {test_accuracy}')
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    def plot_loss_curve(logging_info, val_accuracies):
        """
        Plots the loss and accuracy curves.
        Args:
            logging_info: An arbitrary object containing logging information.
        """
        loss_curve = logging_info['train_loss']

        acc_curve = val_accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(loss_curve)
        ax1.set_title('Loss curve during training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(acc_curve)
        ax2.set_title('Accuracy during validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.savefig("loss_curve_training_pytorch.jpg", dpi=150)

    plot_loss_curve(logging_info, val_accuracies)