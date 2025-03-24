import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None
        
    def forward(self, input_tensor):
        output_tensor = 1 / (1 + np.exp((-1) * input_tensor))
        
        # save activation for backward propagation.
        self.activations = np.copy(output_tensor)
        
        return output_tensor
    
    def backward(self, error_tensor):
        
        output_tensor = self.activations * (1 - self.activations) * error_tensor
        
        return output_tensor