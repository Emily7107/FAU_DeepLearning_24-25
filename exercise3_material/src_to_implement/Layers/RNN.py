import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import copy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True
        self.hidden_state = np.zeros(self.hidden_size)
        self._memorize = False
        self.fc_h = FullyConnected(self.hidden_size + self.input_size ,self.hidden_size)
        self.fc_hy = FullyConnected(self.hidden_size,self.output_size)
        self.weights = self.fc_h.weights
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self._optimizer = None

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        time = input_tensor.shape[0]
        self.tanh_activation = []
        self.sig_activation = []
        self.input_tensor_h = []
        self.input_tensor_hy = []

        if not self._memorize: 
            self.hidden_state = np.zeros(self.hidden_size)

        # go through every time state.
        for t in range(time):
            x = input_tensor[t]
            x_tilde = np.concatenate((self.hidden_state, x), axis = 0) # bias will be add inside the FC layer.
            x_tilde = np.expand_dims(x_tilde,axis=0)
            hidden_t = self.tanh.forward(self.fc_h.forward(x_tilde))
            output_temp = self.sigmoid.forward(self.fc_hy.forward(hidden_t))
            self.hidden_state = np.squeeze(hidden_t) # save hidden_state for next iteration

            # save the state for backward propagation
            self.tanh_activation.append(self.tanh.activations)
            self.sig_activation.append(self.sigmoid.activations)
            self.input_tensor_h.append(self.fc_h.input_tensor)
            self.input_tensor_hy.append(self.fc_hy.input_tensor)

            if t == 0:
                output_tensor = output_temp
            else:
                output_tensor = np.append(output_tensor,output_temp,axis=0)
        
        return output_tensor

    def backward(self, error_tensor):
        
        time = error_tensor.shape[0]
        error_x = np.zeros_like(self.input_tensor)
        error_h = 0
        self.gradient_weights_h = np.zeros_like(self.fc_h.weights)
        self.gradient_weights_hy = np.zeros_like(self.fc_hy.weights)

        for t in reversed(range(time)):
            
            yt = np.expand_dims(error_tensor[t],axis=0)

            # set the activation and input_tensor for each state
            self.sigmoid.activations = self.sig_activation[t]
            self.tanh.activations = self.tanh_activation[t]
            self.fc_h.input_tensor = self.input_tensor_h[t]
            self.fc_hy.input_tensor = self.input_tensor_hy[t]

            # Do backward propagation by using the backward method in each layer
            error_sigmoid = self.sigmoid.backward(yt)
            error_hy = self.fc_hy.backward(error_sigmoid)

            # Add hidden tensor from previous state
            error_hy += error_h

            error_tanh = self.tanh.backward(error_hy)
            error_wh = self.fc_h.backward(error_tanh)

            # split error_wh to h part and x part
            error_h = error_wh[:,:self.hidden_size]
            error_x[t] = np.squeeze(error_wh[:, self.hidden_size:])

            # accumulate gradient weights
            self.gradient_weights_h += self.fc_h.gradient_weights
            self.gradient_weights_hy += self.fc_hy.gradient_weights

        if self._optimizer is not None:
            self.fc_h.weights = self._optimizer.calculate_update(self.fc_h.weights, self.gradient_weights_h)
            self.fc_hy.weights = self._optimizer.calculate_update(self.fc_hy.weights, self.gradient_weights_hy)
            
        return error_x

    def initialize(self,weights_initializer, bias_initializer):
        self.fc_h.initialize(weights_initializer,bias_initializer)
        self.fc_hy.initialize(weights_initializer,bias_initializer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memory):
        self._memorize = memory

    @property
    def weights(self):
        return self.fc_h.weights
    
    @weights.setter
    def weights(self, weights):
        self.fc_h.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_h
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)