import numpy as np
import matplotlib.pyplot as plt

class Checker:
    
    def __init__(self, resolution, tile_size):
        if resolution % tile_size != 0:
            raise ValueError(
                "Resolution must be divisible by tile size without remainder.")
        
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
    
    def draw(self):
        
        # determine repeat pattern.
        repeat = int(self.resolution / (2 * self.tile_size))
        
        # set up black and white blocks.
        black = np.zeros((self.tile_size,self.tile_size))
        white = np.ones((self.tile_size,self.tile_size))
        
        # combine blocks to make a checkerboard.
        style_1= np.concatenate((black,white), axis= 0)
        style_2= np.concatenate((white,black), axis=0)
        block = np.concatenate((style_1,style_2), axis= 1)
        
        self.output = np.tile(block,(repeat,repeat))
        
        return np.copy(self.output)
        
    def show(self):
        
        plt.imshow(self.output,cmap= 'gray')
        plt.show()

class Circle:
    
    def __init__(self, resolution, radius, position):
        
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
    
    def draw(self):
        
        # crate an array use for drawing the circle later.
        paper = np.zeros((self.resolution,self.resolution))
        
        # crate coordinate.
        coordinate = np.linspace(0,self.resolution,self.resolution, dtype=int)
        x , y = np.meshgrid(coordinate,coordinate)
        
        # set up condition to select the elements we want.
        distance = np.sqrt((x-self.position[0])**2+(y-self.position[1])**2)
        condlist = [distance <=self.radius , distance > self.radius]
        choicelist = [ paper + 1 , paper]
        
        self.output = np.select( condlist , choicelist)
        
        return np.copy(self.output)
    
    def show(self):
        if self.output is None:
            raise ValueError(
                "Checkerboard pattern has not been drawn yet.")
            
        plt.imshow(self.output, cmap='gray')
        plt.show()
        
class Spectrum:
    
    def __init__(self, resolution):
        
        self.resolution = resolution
        self.output = None
        
    def draw(self):
        
        x = np.linspace(0,1,self.resolution)
        y = np.linspace(0,1,self.resolution)
        z = np.linspace(1,0,self.resolution)
        
        R , G = np.meshgrid(x,y)
        B , _ = np.meshgrid(z,y)
        
        intensity = np.stack((R,G,B),axis=2)

        self.output = np.round(intensity, 2)

        return np.copy(self.output)
    
    def show(self):
        
        plt.imshow(self.output)
        plt.show()
        
        
        