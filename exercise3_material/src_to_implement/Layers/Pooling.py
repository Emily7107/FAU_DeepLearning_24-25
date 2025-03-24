import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    
    def __init__(self,stride_shape, pooling_shape):
        super().__init__()
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape[0],stride_shape[0])
        else:
            self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_pos = None
        self.input_tensor = None
        self.output_tensor = None
    
    def forward(self, input_tensor):
        # input tensor shape (batch, channel, y, x)
        self.input_tensor = input_tensor
        
        batch = input_tensor.shape[0]
        channel = input_tensor.shape[1]
        row = input_tensor.shape[2] - self.pooling_shape[0] + 1
        col = input_tensor.shape[3] - self.pooling_shape[1] + 1
        outShape_y = int(np.ceil((row) / self.stride_shape[0]))
        outShape_x = int(np.ceil((col) / self.stride_shape[1]))
        output_tensor = np.zeros((batch, channel, outShape_y, outShape_x))
        self.max_pos = [ ]
        
        for b in range(input_tensor.shape[0]):
            temp_tensor = np.zeros((channel, row, col))
            
            for c in range(channel): 
                temp_pos = [ ] 
                for y in range(row):
                    for x in range (col):
                        # calculate max pooling
                        temp_tensor[c][y][x] = np.max(input_tensor[b][c][y: (y+ self.pooling_shape[0]), x: (x+self.pooling_shape[1])])
                        
                        # save the index of max element
                        idx_shape = input_tensor[b][c][y: (y+ self.pooling_shape[0]), x: (x+self.pooling_shape[1])].shape
                        idx = np.unravel_index(np.argmax(input_tensor[b][c][y: (y+ self.pooling_shape[0]), x: (x+self.pooling_shape[1])]),idx_shape)
                        temp_pos.append((b,c,idx[0] + y, idx[1] + x))
                
                for y in range(outShape_y):
                    for x in range(outShape_x):
                        output_tensor[b][c][y][x] = temp_tensor[c][y * self.stride_shape[0]][x * self.stride_shape[1]]
                        self.max_pos.append(temp_pos[(y * self.stride_shape[0]) * col + x * self.stride_shape[1]])

        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        
        output_tensor = np.zeros_like(self.input_tensor)
        batch , channel, row, col = error_tensor.shape
        for b in range(batch):            
            for c in range(channel):
                for y in range(row):
                    for x in range (col):
                        num = x + col * y + c * col * row + b * col * row * channel
                        output_tensor[self.max_pos[num]] += error_tensor[b,c,y,x]
                        
        return output_tensor