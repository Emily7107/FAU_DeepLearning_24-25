import numpy as np

class CrossEntropyLoss:
    
    def __init__(self):
        self.prediction_tensor = None
    
    def forward(self,prediction_tensor,label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = np.sum(label_tensor * (-1)*np.log(prediction_tensor + np.finfo(float).eps))
        
        return loss
    
    def backward(self,label_tensor):
        error_tensor = (-1) * label_tensor / (self.prediction_tensor + np.finfo(float).eps)
        
        return error_tensor
