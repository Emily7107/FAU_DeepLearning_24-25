import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import skimage as ski
import glob
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        
        self.file_path = os.path.join(file_path, '*.npy')
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        self.count_next = 0
        self.epoch = 0 
        
        # load data
        images_temp = [ ]
        labels_temp = [ ]
        
        with open(self.label_path) as f:
            label = json.load(f)
        
        files = glob.glob(self.file_path)
        
        for file in files:
            file = os.path.normpath(file)
            img = np.load(file)
            img = ski.transform.resize( img, self.image_size)
            images_temp.append(img)
            
            temp = os.path.basename(file)
            num = temp.split('.')[0]
            labels_temp.append(label[num])
        
        images_temp = np.array(images_temp)
        labels_temp = np.array(labels_temp)
        
        # complete data
        if len(images_temp) % self.batch_size != 0:
            add = self.batch_size - (len(images_temp) % self.batch_size)
            images_temp = np.concatenate((images_temp,images_temp[0:add,:,:,:]),axis=0)
            labels_temp = np.concatenate((labels_temp,labels_temp[0:add]),axis=0)
        
        self.images = images_temp
        self.labels = labels_temp

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        if self.count_next >= len(self.images):
            self.count_next = 0
            self.epoch +=1
        
        if self.shuffle == True and self.count_next == 0:
            shuffle_index = np.random.permutation(len(self.images))
            self.images = self.images[shuffle_index]
            self.labels = self.labels[shuffle_index]
        
        i = self.count_next
        images = np.copy(self.images[i:i+self.batch_size,:,:,:])
        labels = np.copy(self.labels[i:i+self.batch_size])
        
        if self.rotation == True:
            for i in range(len(images)):
                images[i] = np.rot90(images[i],k=random.randrange(3))
                
        if self.mirroring == True:
            for i in range(len(images)):
                images[i] = np.flip(images[i], 0 )
        
        self.count_next += self.batch_size
            
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        
        if self.rotation == True:
                img = np.rot90(img,k=random.randrange(3))
        
        if self.mirroring == True:
                img = np.flip(img, 0 )

        return img

    def current_epoch(self):
        # return the current epoch number
        
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        
        return self.class_dict.get(x, "Unknown")
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        object = self.next()
        img = object[0]
        label= object[1]
        
        for i in range(len(img)):
            plt.gcf().set_size_inches(8,14)
            plot=plt.subplot(4,3,1+i)
            plot.imshow(img[i] , cmap='binary')
            plot.set_title(self.class_name(label[i]), fontsize = 12)
            plot.set_xticks([])
            plot.set_yticks([])

        plt.show()
