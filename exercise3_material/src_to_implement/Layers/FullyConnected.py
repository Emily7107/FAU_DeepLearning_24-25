import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0.0,1.0,size=(input_size + 1 ,output_size))
        self._optimizer = None
        self._gradient_weights = None
        
    def initialize(self, weight_initializer , bias_initializer):
        weights = weight_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        bias = bias_initializer.initialize((1,self.output_size),self.input_size,self.output_size)
        
        self.weights = np.concatenate((weights, bias), axis = 0 )
    
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        
        # add bias
        bias = np.ones((batch_size,1))
        input_tensor_bias = np.hstack((input_tensor,bias))
        
        # forward propagation calculation
        output_tensor = input_tensor_bias @ self.weights
        self.input_tensor = input_tensor_bias
        
        return np.copy(output_tensor)
    
            
    def backward(self, error_tensor):
        # self.input_tensor is the output of previous layer, e.g. the input_tensor of forward propagation.
        self._gradient_weights = self.input_tensor.T @ error_tensor
        
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights , self._gradient_weights)
        
        # drop bias
        weight_drop = self.weights[:-1, :]
        
        # backward propagation calculation
        output_tensor = error_tensor @ weight_drop.T
        
        return np.copy(output_tensor)
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def gradient_weights(self):
        return np.copy(self._gradient_weights)