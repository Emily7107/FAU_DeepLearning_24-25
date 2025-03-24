import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    
    def __init__(self):
        super().__init__()
        self.prediction = None
        
    def forward(self,input_tensor):
        numeric = input_tensor.T-np.amax(input_tensor, axis = 1)
        exp_x = np.exp(numeric.T)
        prediction = exp_x.T/np.sum(exp_x , axis= 1)
        self.prediction = prediction.T
        
        return np.copy(self.prediction)
    
    def backward(self, error_tensor):
        temp = error_tensor.T - np.sum(error_tensor * self.prediction, axis=1)
        output_error = self.prediction * temp.T
        
        return output_error
