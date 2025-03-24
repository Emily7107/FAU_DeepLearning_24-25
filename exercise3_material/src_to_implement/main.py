import unittest
try:
    LSTM_TEST = True
    from Layers import *
except BaseException as e:
    if str(e)[-6:] == "'LSTM'":
        LSTM_TEST = False
    else:
        raise e
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import argparse
import tabulate


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
iterations = 100

def _perform_test(optimizer, iterations, description, dropout, batch_norm):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          Initializers.He(),
                                          Initializers.Constant(0.1))
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 150
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNormalization(1))

        cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
        net.append_layer(cl_1)
        cl_1_output_shape = (num_kernels, *input_image_shape[1:])

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNormalization(num_kernels))

        net.append_layer(ReLU.ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        net.append_layer(Flatten.Flatten())

        fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNormalization(fcl_1_input_size//2))

        if dropout:
            net.append_layer(Dropout.Dropout(0.3))

        net.append_layer(ReLU.ReLU())

        fcl_2 = FullyConnected.FullyConnected(int(fcl_1_input_size / 2), int(fcl_1_input_size / 3))
        net.append_layer(fcl_2)

        net.append_layer(ReLU.ReLU())

        fcl_3 = FullyConnected.FullyConnected(int(fcl_1_input_size / 3), categories)
        net.append_layer(fcl_3)

        net.append_layer(SoftMax.SoftMax())

        net.train(iterations)
        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        # with open(self.log, 'a') as f:
        #     print('On the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.), file=f)
        # print('\nOn the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.))
        # self.assertGreater(accuracy, 0.3)

sgd_with_l2 = Optimizers.Adam(1e-2, 0.98, 0.999)
sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
_perform_test(sgd_with_l2, iterations, 'Batch_norm and L2', False, True)