import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.drop = None
    
    def forward(self, input_tensor):
        output_tensor = input_tensor
        
        # if we are at training phase, randomly dropout nodes base on the probability.
        if self.testing_phase == False:
            # there are p% chance to be 1, and 1-p% chance to be 0.
            self.drop = np.random.binomial(1, self.probability, input_tensor.shape)
            output_tensor = (input_tensor * self.drop ) / self.probability
            
        return output_tensor
    
    def backward(self, error_tensor):
        # also dropout node when backward propagation.
        output_tensor = (error_tensor * self.drop) / self.probability
        
        return output_tensor
    
