import numpy as np
from Layers.Base import BaseLayer
from scipy.signal import convolve,correlate
import copy

class Conv(BaseLayer):
    
    def __init__(self, stride_shape, convolution_shape, num_kernels : int):
        super().__init__()
        self.trainable = True
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape[0],stride_shape[0])
        else:
            self.stride_shape = stride_shape
        
        if len(convolution_shape) ==2:
            self.convolution_shape = (*convolution_shape,1) # 1D: [c,m] 2D: [c,m,n] c is input channels, m & n are spatial extent
        else:
            self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0,1,size = (num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0,1,size = num_kernels)
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.input_tensor = None

    def forward(self, input_tensor):
        
        # if input_tensor is 1D, reshape the tensor
        is_1D = False
        if len(input_tensor.shape) ==3:
            input_tensor = input_tensor.reshape((*input_tensor.shape,1))
            is_1D = True
            
        # save input shape for backward 
        self.input_tensor = input_tensor
        batch = input_tensor.shape[0]
        channel = input_tensor.shape[1]
        
        outShape_y = int(np.ceil(input_tensor.shape[-2] / self.stride_shape[0]))
        outShape_x = int(np.ceil(input_tensor.shape[-1] / self.stride_shape[1]))
        output_tensor = np.zeros((input_tensor.shape[0],self.num_kernels,outShape_y,outShape_x))
        
        # CNN calculation
        for b in range(batch):
            for k in range(self.num_kernels):
                
                temp_tensor = np.zeros((input_tensor.shape[-2],input_tensor.shape[-1]))
                for c in range(channel):
                    temp_tensor += correlate(input_tensor[b][c][:], self.weights[k][c][:], mode='same')

                # add bias (element-wise)
                temp_tensor += self.bias[k]
                
                # deal with stride
                for y in range(output_tensor.shape[-2]):
                    for x in range(output_tensor.shape[-1]):
                        output_tensor[b][k][y][x] = temp_tensor[y * self.stride_shape[0]][x * self.stride_shape[1]]
                        
        # if is_1D is True, squeeze output_tensor to ariginal size
        if is_1D:
            output_tensor = np.squeeze(output_tensor,axis=-1)
        return output_tensor
        
    def backward(self, error_tensor):
        
        is_1D = False
        if len(error_tensor.shape) ==3:
            error_tensor = error_tensor.reshape((*error_tensor.shape,1))
            is_1D = True
        
        # error_tensor.shape = (batch, num_kernels, y, x)
        # num_kernels = channels
        batch = error_tensor.shape[0]
        channel = self.num_kernels
        
        # weight rearrange
        weights_T = self.weights.transpose(1,0,2,3)
        
        output_tensor = np.zeros_like(self.input_tensor)
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        
        # claculate backward CNN
        for b in range(batch):
            for k in range(self.input_tensor.shape[1]):
                
                # deal with stride unsampling
                temp_tensor = np.zeros((error_tensor.shape[1],self.input_tensor.shape[-2],self.input_tensor.shape[-1]))             
                for c in range(channel):
                    for y in range(error_tensor.shape[-2]):
                        for x in range(error_tensor.shape[-1]):
                            temp_tensor[c][y * self.stride_shape[0]][x * self.stride_shape[1]] = error_tensor[b][c][y][x]
                    
                    # using convolve here, since we used correlate in forward
                    output_tensor[b][k][:] += convolve(temp_tensor[c][:], weights_T[k][c][:], mode='same')
                    
            # calculating gradience bias
            for k in range(self.num_kernels):
                self._gradient_bias[k] += np.sum(error_tensor[b, k, :])
            
            # determine padding size
            pad_y = int(np.floor(self.convolution_shape[-2] / 2))
            pad_x = int(np.floor(self.convolution_shape[-1] / 2))
            
            if self.convolution_shape[-2] % 2 ==0:
                padShape_y = (pad_y-1,pad_y)
            else:
                padShape_y = (pad_y,pad_y)
                
            if self.convolution_shape[-1] % 2 ==0:
                padShape_x = (pad_x-1,pad_x)
            else:
                padShape_x = (pad_x,pad_x)
            
            padding_tensor = np.pad(self.input_tensor,[(0,0),(0,0),padShape_y,padShape_x],mode='constant',constant_values=0)
            
            # calculate gradient weight
            for k in range(self.num_kernels):
                for c in range(padding_tensor.shape[1]):
                    self._gradient_weights[k][c][:] +=correlate(padding_tensor[b][c][:], temp_tensor[k][:], mode='valid')
        
        if self._optimizer_weights != None:
            self.weights = self._optimizer_weights.calculate_update(self.weights , self._gradient_weights)
        
        if self._optimizer_bias != None:
            self.bias = self._optimizer_bias.calculate_update(self.bias , self._gradient_bias)
        
        if is_1D:
            output_tensor = np.squeeze(output_tensor,axis=-1)
        return output_tensor
        
    
    def initialize(self,weight_initializer,bias_initializer):
        
        fan_in = self.convolution_shape[0] * self.convolution_shape[-1] * self.convolution_shape[-2]
        fan_out = self.num_kernels * self.convolution_shape[-1] * self.convolution_shape[-2]
        
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
    
    @property
    def optimizer(self):
        return self._optimizer_weights
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)
    
    @property
    def gradient_weights(self):
        return np.copy(self._gradient_weights)
    
    @property
    def gradient_bias(self):
        return np.copy(self._gradient_bias)