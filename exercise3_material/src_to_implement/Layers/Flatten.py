import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    
    def __init__(self):
        super().__init__()
        self.shape = None
    
    def forward(self, input_tensor):
        output_tensor = input_tensor.reshape((input_tensor.shape[0],-1))
        self.shape = input_tensor.shape
        
        return np.copy(output_tensor)
    
    def backward(self, error_tensor):
        output_tensor = error_tensor.reshape(self.shape)
        
        return np.copy(output_tensor)