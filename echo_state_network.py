# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:07:54 2015

@author: Adam Conkey

This is an implementation of an Echo State Network to be used for semantic classification
of English sentences. The goal is to vectorize sentences using word2vec and feed them into
the ESN to train it to recognize the desired semantic quality (e.g. whether or not a 
sentence contains a causal relation). The intuition is that ESNs are good at capturing
historical information in a signal, and if a sentence can be treated as a signal, perhaps
the ESN can pick up on the temporal and spatial dependencies existing in the sentence. 

This endeavor was initiated with the intent of assisting Dr. Peter Hastings of DePaul University
in his research regarding the application of machine learning techniques to infer causal 
structure in student essays.

"""
import numpy as np
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge
import random

def run_simulation(esn):
    """
    Train the output weights of the provided Echo State Network with the network's associated
    training set and then drive the network with its associated testing set.

    :param esn: Echo State Network to be trained and tested
    :return: computed accuracy as ratio of correct classifications / total test examples
    """
    # compute reservoir states over train duration:
    drive_network_train(esn)

    # compute the output weights using Ridge Regression:
    clf = Ridge()  # play with parameters ???
    clf.fit(esn.x_r, esn.x_target_train)
    esn.w_out = clf.coef_

    # drive with test input:
    targets, outputs, accuracy = drive_network_test(esn)
    print 'Accuracy:'
    print accuracy
    print 'Targets:'
    print targets
    print 'Outputs:'
    print outputs

def drive_network_train(esn):
    """
    Drives the reservoir with training input and stores the reservoir states over time.

    :param esn: Echo State Network to train with
    """
    for instance in esn.train_data:
        x_in = np.vstack((np.zeros((1, esn.n_in)), instance[0]))
        x_target = np.vstack((np.zeros((1, esn.n_out)),
                              np.tile(instance[1], (instance[0].shape[0], 1))))  # repeat for each time step
        duration = x_in.shape[0]
        x_instance = np.zeros((duration, esn.n_r))
        for i in range(1, duration + 1):  # +1 because loop range is shifted by initial zero vectors
            print 'x in [i]:'
            print x_in[i]
            print 'weights:'
            print esn.w_in.shape
            x_instance[i] = np.tanh(x_in[i] * esn.w_in)
                                    + (x_instance[i - 1] * esn.w_r)
                                    + (x_target[i - 1] * esn.w_fb))
            x_instance[i] = (1 - esn.alpha) * x_instance[i - 1] + esn.alpha * x_instance[i]
            "might need to initialize esn.x_r properly before stacking; done in class instance initialization?"
        np.vstack((esn.x_r, x_instance[1:]))  # add instance activation except initial zero activation
        np.vstack((esn.x_target_train, x_target[1:]))  # same with targets


def drive_network_test(esn):
    """
    Drive the reservoir with novel input and use the trained output weights to compute output values.

    :param esn: Echo State Network to test with
    """
    targets = np.zeros((esn.test_data.shape[0], esn.n_out))
    outputs = np.zeros(targets.shape)
    for j in range(esn.test_data):
        instance = esn.test_data[j]
        x_in = np.vstack((np.zeros((1, esn.n_in)), instance[0]))
        targets[j] = instance[1]  # only a single vector for testing to match target classification for sentence
        duration = x_in.shape[0]
        x_instance = np.zeros((duration, esn.n_r))
        x_out = np.zeros((duration, esn.n_out))
        for i in range(1, duration + 1):
            x_instance[i] = np.tanh((x_in[i] * esn.w_in)
                                    + (x_instance[i - 1] * esn.w_r)
                                    + (x_out[i - 1] * esn.w_fb))
            x_instance[i] = (1 - esn.alpha) * x_instance[i - 1] + esn.alpha * x_instance[i]
            x_out[i] = sigmoid(x_instance[i].dot(esn.w_out.T))

            # will maybe want a more efficient way of doing this:
            max_index = x_out[i].argmax()
            x_out[i] = np.zeros(x_out[i].shape)
            x_out[i][max_index] = 1

        # for now I am just taking the instance classification to be the classification at the end of
        # instance chain. should print out activations and see how they fare, may want to have an
        # averaging window
        outputs[j] = x_out[duration]

    accuracy = compute_accuracy(targets, outputs)
    return targets, outputs, accuracy

def compute_accuracy(expected, actual):
    """
    Simple computation of accuracy taking the number of correct over total.

    Keyword arguments:
    expected    -- array of 1s and 0s, expected values
    actual      -- array of 1s and 0s, actual values, same size as expected
    """
    total = float(expected.shape[0])
    n_correct = np.sum(np.multiply(expected, actual))
    return n_correct / total

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    run_simulation()

class EchoStateNetwork:
    """
    An Echo State Network with an input layer, reservoir, and output layer with associated connections.

    Attributes:
        n_in            number of input units
        n_r             number of reservoir units
        n_out           number of readout units

        n_train         number of training instances
        n_test          number of testing instances

        w_in		weights from input to reservoir			    : n_r   x n_in + 1
        w_r		weights internal to reservoir                       : n_r   x n_r
        w_out		weights from reservoir to readout                   : n_r   x n_out
        w_fb            weights from readout back into reservoir            : n_out x n_r

        test_size       percentage of input data that is the testing set
            		NOTE: a train set will be constructed as complement of test set

        x_in_train      activations of input nodes over train time          : t_train x n_in
        x_in_test       activations of input nodes over test time           : t_test x n_in
        x_target_train  target activations of output nodes over train time  : t_train x n_out
        x_target_test   target activations of output nodes over test time   : t_test x n_out
        x_r             activations of reservoir nodes over time            : t_train x n_r
        x_out           activations of output nodes over time               : t_train x n_out
        x_target        target activations for training                     : t_train x n_out

        scale_in        scale factor of input weights
        scale_r		    scale factor of reservoir weights
        scale_fb        scale factor of feedback weights

        density_in      density coefficient for input-reservoir
        density_r       density coefficient for reservoir
        density_fb      density coefficient for out-reservoir

        alpha	        leaking rate of reservoir units
        rho		        desired spectral radius of reservoir

        seed 	        seed for RandomState
        rs              RandomState for random number generation

    """
    def __init__(self, data, seed=123, n_r=100, density_in=1.0, density_r=0.1,
                 density_fb=1.0, scale_in=1.0, scale_r=1.0, scale_fb=1.0, alpha=0.9,
                 rho=0.9, w_out=[], test_size=0.2, x_r=[], x_out=[]):
        self.data = self.add_bias(data)
        self.n_in = self.data[0][0].shape[1]
        self.n_out = self.data[0][1].shape[1]
        self.seed = seed
        self.rs = RandomState(self.seed)
        self.n_r = n_r
        self.density_in = density_in
        self.density_r = density_r
        self.density_fb = density_fb
        self.scale_in = scale_in
        self.scale_r = scale_r
        self.scale_fb = scale_fb
        self.rho = rho
        self.alpha = alpha
        self.w_in = self.initialize_weights(self.n_in + 1, self.n_r, self.density_in, self.rs, self.scale_in)
        self.w_r = self.initialize_reservoir(self.n_r, self.density_r, self.rs, self.scale_r, self.rho)
        self.w_fb = self.initialize_weights(self.n_out, self.n_r, self.density_fb, self.rs, self.scale_fb)
        self.w_out = w_out
        self.test_size = test_size
        self.train_data, self.test_data = self.train_test_split(self.data, self.test_size)
        self.x_r = x_r
        self.x_out = x_out

    @staticmethod
    def initialize_weights(n_rows, n_cols, density, randomstate, scale):
        """
        Initialize a sparse random matrix of weights with dimensions
        n_rows x n_cols and specified density in range [-scale, scale].

        Keyword arguments:
        n_rows      -- number of rows
        n_cols      -- number of columns
        density     -- density of connections
        randomstate -- RandomState object for random number generation
        scale       -- absolute value of minimum/maximum weight value
        """
        weights = sparse.rand(n_rows, n_cols, density, random_state=randomstate)
        weights = 2 * scale * weights - scale * weights.ceil()
        return weights

    @staticmethod
    def initialize_reservoir(n_units, density, randomstate, scale, spec_rad):
        """
        Initialize a sparse random reservoir as a square matrix representing connections among
        the n_units neurons with connections having specified density in range [-scale, scale].

        The weights are generated until they achieve a spectral radius of at least 0.01;
        due to the iterative nature of scipy.sparse.linalg.eigs, values under this threshold
        are unstable and do not produce consistent results over time.

        Keyword arguments:
        n_units     -- number of reservoir nodes
        density     -- density of connections (default 0.1)
        randomstate -- RandomState object for random number generation (default RandomState(1))
        scale       -- absolute value of minimum/maximum weight value (default 1.0)
        spec_rad    -- desired spectral radius to scale to (default 1.0)
        """
        while True:
            weights = EchoStateNetwork.initialize_weights(n_units, n_units, density, randomstate, scale)
            if max(abs(eigs(weights)[0])) >= 0.01:
                break

        weights = EchoStateNetwork.scale_spectral_radius(weights, spec_rad)
        return weights

    @staticmethod
    def scale_spectral_radius(weights, spec_rad):
        """
        Scales the specified weight matrix to have the desired spectral radius.

        Keyword arguments:
        weights     -- weight array to scale
        spec_rad    -- desired spectral radius to scale to (default 1.0)
        """
        weights = spec_rad * (weights / max(abs(eigs(weights)[0])))
        return weights

    @staticmethod
    def create_pseudo_signal(data, reps):
        """
        Create a pseudo-signal from a data set by repeating each row the specified number of times.

        :param data: basis of pseudo-signal
        :param reps: number of times each data element is to be repeated
        :return: pseudo-signal of original data
        """
        signal = np.tile(data[0], (reps, 1))
        for i in range(1, data.shape[0]):
            signal = np.vstack((signal, np.tile(data[i], (reps, 1))))
        return signal

    @staticmethod
    def add_bias(data):
        biased = data
        for x in biased:
            ones = np.ones((x[0].shape[0], 1))
            np.concatenate((ones, x[0]), axis=1)
            print x
        return biased

    @staticmethod
    def train_test_split(data, test_size):
        random.shuffle(data)

        split_index = int(test_size * len(data))
        test_data = data[:split_index]
        train_data = data[split_index:]

        return train_data, test_data
