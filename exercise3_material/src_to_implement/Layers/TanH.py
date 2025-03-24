import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None
        
    def forward(self, input_tensor):
        output_tensor = np.tanh(input_tensor)
        
        # save activation for backward propagation.
        self.activations = np.copy(output_tensor)
        
        return output_tensor
    
    def backward(self, error_tensor):
        
        output_tensor = (1 - np.square(self.activations)) * error_tensor
        
        return output_tensor