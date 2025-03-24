import numpy as np

class Constant:
    
    def __init__(self, init_value = 0.1):
        self.init_value = init_value
    
    def initialize(self, weight_shape ,fan_in = None , fan_out = None):
        
        weight = np.ones(shape=(weight_shape)) * self.init_value
        
        return weight
    
class UniformRandom:
    
    def __init__(self):
        pass
    
    def initialize(self, weight_shape, fan_in = None, fan_out = None):
        
        weight = np.random.uniform(0.0,1.0,size=(weight_shape))
        
        return weight
    
class Xavier:
    
    def __init__(self):
        pass
    
    def initialize(self, weight_shape, fan_in = None, fan_out = None):
        
        a = np.sqrt(2.0/(fan_out+fan_in))
        weight = np.random.normal(0.0,a,size = weight_shape)
        
        return weight

class He:
    
    def __init__(self):
        pass
    
    def initialize(self, weight_shape, fan_in = None, fan_out = None):
        
        a = np.sqrt(2.0/ fan_in)
        weight = np.random.normal(0.0,a,size = weight_shape)
        
        return weight
