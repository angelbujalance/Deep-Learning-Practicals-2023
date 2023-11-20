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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.

        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        # TODO input layer
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        if input_layer:
            self.params['weight'] = np.random.normal(0, 1/in_features, (in_features, out_features)) #np.random.normal(0, in_features, out_features) * np.sqrt(1. / float(in_features))
        else:
            self.params['weight'] = np.random.normal(0, 2/in_features, (in_features, out_features)) #np.random.normal(in_features, out_features) * np.sqrt(2. / float(in_features))

        self.params['bias'] = np.zeros(out_features)
        self.grads['weight'] = np.zeros((in_features, out_features))
        self.grads['bias'] = np.zeros(out_features)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #self.x = x.reshape(x.shape[0], -1)
        self.x = x
        out = x @ self.params['weight'] + self.params['bias']
        #######################
        # END OF YOUR CODE    #
        #######################

        self.shape = x.shape

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = self.x.T @ dout
        self.grads['bias'] = np.ones(dout.shape[0]).T @ dout
        #print(dout.shape)
        #print(self.params['weight'].T.shape)
        dx = dout @ self.params['weight'].T
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        self.shape = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = x
        out = np.where(x > 0, x, (np.exp(x)-1))
        grad = np.where(x > 0, 1, np.exp(x))
        self.grad = grad
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # activation module derivative
        dx = self.grad * dout
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        self.grad = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        b = x.max(axis=1, keepdims=True)#.reshape(-1, 1)
        y = np.exp(x - b)
        out = (y / y.sum(axis=1, keepdims=True))#.reshape(-1, 1) # / (y / y.sum() + 1)
        self.out = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        dL/dZ = Y * (dL/dY - (dL/dY) * Y)11^T)
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = self.out * ((dout - (dout * self.out) @ np.ones((self.out.shape[1], self.out.shape[1]))))
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.out = None #np.zeros(self.grads['weight'].size)
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #epsilon = 1e-15
        #x = np.clip(x, epsilon, 1. - epsilon)
        #I_matrix = np.eye(x.shape[1])
        #y = I_matrix[y] if y.shape != x.shape else y
        ##print(x.shape); print(y.shape)
        #out = -np.sum(y * np.log(x), axis=1).mean()
        if y.shape != x.shape:
            y = np.eye(x.shape[1])[y]
        out = 1 / x.shape[0] * np.sum(-y * np.log(x))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #I_matrix = np.eye(x.shape[1])
        #y = I_matrix[y]
        #dx = -np.divide(y, x)[:] / x.shape[0]
        if y.shape != x.shape:
            y = np.eye(x.shape[1])[y]
        dx = -1 / x.shape[0] * np.divide(y, x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx