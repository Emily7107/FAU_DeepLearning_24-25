import unittest
from Layers import *
from Optimization import *
import numpy as np
from scipy import stats
# from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import tabulate
import argparse

class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)

plot = False
directory = 'plots/'
log = 'log.txt'

net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
categories = 3
input_size = 4
net.data_layer = Helpers.IrisData(100)
net.loss_layer = Loss.CrossEntropyLoss()
fcl_1 = FullyConnected.FullyConnected(input_size, categories)
net.append_layer(fcl_1)
net.append_layer(ReLU.ReLU())
fcl_2 = FullyConnected.FullyConnected(categories, categories)
net.append_layer(fcl_2)
net.append_layer(SoftMax.SoftMax())

net.train(4000)
if plot:
    fig = plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
    plt.plot(net.loss, '-x')
    fig.savefig(os.path.join(directory, "TestNeuralNetwork2.pdf"), transparent=True, bbox_inches='tight',
                pad_inches=0)

data, labels = net.data_layer.get_test_set()

results = net.test(data)

accuracy = Helpers.calculate_accuracy(results, labels)
with open(log, 'a') as f:
    print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)