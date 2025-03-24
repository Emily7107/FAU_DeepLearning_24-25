import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import copy

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self.batch_mean = None
        self.batch_var = None
        self.moving_mean = None
        self.moving_var = None
        self.decay = 0.8
        self._optimizer= None
        self._gradient_weights = None
        self._gradient_bias = None
        
    def initialize(self, weight_initializer = None , bias_initializer = None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    def forward(self, input_tensor):
        
        is_conv = False
        # reformat image-like tensor.
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
            is_conv = True
        
        self.input_tensor = input_tensor
        
        # at training time.
        if self.testing_phase != True:
            self.batch_mean = np.mean(input_tensor, axis=0)
            self.batch_var = np.var(input_tensor,axis=0) # the square of standard deviation is the variance
            
            self.input_tensor_tilder = np.subtract(input_tensor, self.batch_mean) / np.sqrt(self.batch_var+ np.finfo(float).eps)
        
        # at testing time.
        else:
            # implement the moving average estimation of training set mean and variance.
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.batch_mean)
            if self.moving_var is None:
                self.moving_var = copy.deepcopy(self.batch_var)
            
            self.moving_mean = self.decay * self.moving_mean + (1-self.decay) * self.batch_mean
            self.moving_var = self.decay * self.moving_var + (1-self.decay) * self.batch_var
            
            self.input_tensor_tilder = np.subtract(input_tensor, self.moving_mean) / np.sqrt(self.moving_var+ np.finfo(float).eps)
        
        output_tensor = self.weights * self.input_tensor_tilder + self.bias
        
        # reformate back to image-like tensor.
        if is_conv:
            output_tensor = self.reformat(output_tensor)
            self.input_tensor_tilder = self.reformat(self.input_tensor_tilder)
        
        return output_tensor
    
    def backward(self, error_tensor):
        
        is_conv = False
        if len(error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)
            self.input_tensor_tilder = self.reformat(self.input_tensor_tilder)
            is_conv = True
        
        # calculate gradient weights and gradient bias.
        self._gradient_weights = np.sum(error_tensor * self.input_tensor_tilder, axis=0)
        self._gradient_bias = np.sum(error_tensor,axis=0)
        
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)
        
        # calculate bn gradients by using the method given in Helpers.
        gradient_tensor = compute_bn_gradients(error_tensor,self.input_tensor, self.weights , self.batch_mean,self.batch_var)
        
        if is_conv:
            gradient_tensor = self.reformat(gradient_tensor)
            self.input_tensor_tilder = self.reformat(self.input_tensor_tilder)
        
        return gradient_tensor
    
    def reformat(self,tensor):
        if len(tensor.shape) == 4:
            self.reformat_shape = tensor.shape
            B, H, M, N = tensor.shape
            
            tensor = tensor.reshape((B, H, M * N))
            tensor = tensor.transpose((0, 2, 1))
            tensor = tensor.reshape((B * M * N, H))
        
        else:
            B, H, M, N = self.reformat_shape
            tensor = tensor.reshape((B, M * N, H))
            tensor = tensor.transpose((0, 2, 1))
            tensor = tensor.reshape((B, H, M, N))
            
        return tensor
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        
    @property
    def gradient_weights(self):
        return np.copy(self._gradient_weights)
    
    @property
    def gradient_bias(self):
        return np.copy(self._gradient_bias)