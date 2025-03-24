import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    
    def __init__(self):
        super().__init__()
        self.input_tensor = None
    
    def forward(self, input_tensor):
        
        self.input_tensor = input_tensor
        
        # ReLU: f = max(0,x)
        output_tensor = np.maximum(0,self.input_tensor)
        
        return output_tensor
    
    def backward(self, error_tensor):
        
        # derivative of ReLU: f = 1 if x > 0; f = 0 else
        condlist = [self.input_tensor > 0 , self.input_tensor<=0]
        choicelist = [ 1 , 0]
        derivative = np.select( condlist , choicelist)
        output_error = derivative * error_tensor
        
        return output_error