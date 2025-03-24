import numpy as np

# add base optimizer for including regularizer.
class  Optimizer:
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self,regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    
    def __init__(self, learning_rate : float):
        super().__init__()
        self.learning_rate = learning_rate
        
    def calculate_update(self,weight_tensor,gradient_tensor):
        weight_update = weight_tensor - self.learning_rate * gradient_tensor
        
        if self.regularizer is not None:
            weight_update = weight_update - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        
        return weight_update
    
class SgdWithMomentum(Optimizer):
    
    def __init__(self, learning_rate : float, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(gradient_tensor)
        
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        weight_update = weight_tensor + self.v
        
        if self.regularizer is not None:
            weight_update = weight_update - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        
        return weight_update
    
class Adam(Optimizer):
    
    def __init__(self, learning_rate : float, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.iteration = 1
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        
        if self.v is None:
            self.v = np.zeros_like(gradient_tensor)
        
        if self.r is None:
            self.r = np.zeros_like(gradient_tensor)
        
        self.v = self.mu * self.v + (1 - self.mu ) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor
        
        v_hat = self.v / (1 - self.mu ** self.iteration)
        r_hat = self.r / (1 - self.rho ** self.iteration)
        
        weight_update = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps )
        
        if self.regularizer is not None:
            weight_update = weight_update - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        
        self.iteration += 1
        
        return weight_update

