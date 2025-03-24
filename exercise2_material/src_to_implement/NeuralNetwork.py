import copy

class NeuralNetwork:
    
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        
        # add initializer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        self.prediction = None
        
    def forward(self):
        self.input_tensor , self.label_tensor = self.data_layer.next()
        
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
        
        self.prediction = self.input_tensor
        loss = self.loss_layer.forward(self.input_tensor,self.label_tensor)
            
        return loss
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    
    def append_layer(self,layer):
        if layer.trainable:
            set_optimizer = copy.deepcopy(self.optimizer)
            set_weights_initializer = copy.deepcopy(self.weights_initializer)
            set_bias_initializer = copy.deepcopy(self.bias_initializer)
            layer.optimizer = set_optimizer
            layer.initialize(set_weights_initializer,set_bias_initializer)
        
        self.layers.append(layer)
        
    def train(self, iterations):
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()
    
    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        return input_tensor
