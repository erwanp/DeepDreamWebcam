
# coding: utf-8

# # Deep-Dream Pytorch implementation

# ## Setup instructions
# * Install Conda
# https://www.anaconda.com/download/#macos
# 
# * Install pytorch and torchvision (computer vision for pytorch) package
# 
# `conda install pytorch torchvision -c pytorch`
# 
# https://pytorch.org/

# ## Overview
# * https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html 
# * https://en.wikipedia.org/wiki/DeepDream

# ## References
# * https://github.com/sar-gupta/deep-dream-pytorch

# ### Python imports
# * torch, numpy, torchvision
# * PIL: Python Imaging Library
# * pyplot: plotting image

# In[1]:


import torch
from torchvision import models, transforms
import numpy as np
from matplotlib import pyplot
#get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image, ImageFilter, ImageChops
from time import time


# Program configs

# In[10]:


IMAGE_PATH = 'starry_night.jpg'
CUDA_ENABLED = True

# Deep dream configs
LAYER_ID = 28  # 28 # The layer to maximize the activations through
NUM_ITERATIONS = 3 #5 # Number of iterations to update the input image with the layer's gradient
LR = 0.3  # 0.2

# We downscale the image recursively, apply the deep dream computation, scale up, and then blend with the original image 
# to achieve better result.
NUM_DOWNSCALES = 10 #20
BLEND_ALPHA = 0.3  # 0.6


# DeepDream class: use a vgg16 pretrained model

# In[3]:


class DeepDream:
    def __init__(self, layer_id=LAYER_ID):
        ''' layer_id: int [ 0 - 31]
            type of dreamer
        '''
        self.model = models.vgg16(pretrained=True)
        if CUDA_ENABLED:
            self.model = self.model.cuda()
        self.modules = list(self.model.features.modules())
        
        self.layer_id = layer_id
        
        # vgg16 use 224x224 images
        imgSize = 224
        self.transformMean = [0.485, 0.456, 0.406]
        self.transformStd = [0.229, 0.224, 0.225]
        self.transformNormalise = transforms.Normalize(
            mean=self.transformMean,
            std=self.transformStd
        )
        
        self.transformPreprocess = transforms.Compose([
            transforms.Resize((imgSize, imgSize)),
            transforms.ToTensor(),
            self.transformNormalise
        ])
        
        self.tensorMean = torch.Tensor(self.transformMean)
        if CUDA_ENABLED:
            self.tensorMean = self.tensorMean.cuda()

        self.tensorStd = torch.Tensor(self.transformStd)
        if CUDA_ENABLED:
            self.tensorStd = self.tensorStd.cuda()

    def toImage(self, input):
        return input * self.tensorStd + self.tensorMean


    def load(self, image):
        self.image = image
        
        return self

    def change_id(self, layer_id):
        self.layer_id = layer_id

# ### Main deep dream algorithm
# Input 
# * an input image
# * layerId
# * number of iterations
# 
# Makes a forward pass until layerId, computes the gradient and updates the input image. This is repeated for the given number of iterations.

    def deepDream(self, image, layer, iterations, lr):
        transformed = self.transformPreprocess(image).unsqueeze(0)
        if CUDA_ENABLED:
            transformed = transformed.cuda()
        input = torch.autograd.Variable(transformed, requires_grad=True)
        self.model.zero_grad()
        for _ in range(iterations):
            out = input
            for layerId in range(layer):
                out = self.modules[layerId + 1](out)
            loss = out.norm()
            loss.backward()
            input.data = input.data + lr * input.grad.data

        input = input.data.squeeze()
        input.transpose_(0,1)
        input.transpose_(1,2)
        img = self.toImage(input)
        if CUDA_ENABLED:
            input = np.clip(img.cpu(), 0, 1)
        else:
            input = np.clip(img, 0, 1)
        return Image.fromarray(np.uint8(input*255))


# Recursively apply deepDream at different scales and blend the result images to make the final image looks better.

    def deepDreamRecursive(self, image, layer, iterations, lr, num_downscales):
        if num_downscales > 0:
            # scale down the image
            image_small = image.filter(ImageFilter.GaussianBlur(2))
            small_size = (int(image.size[0]/2), int(image.size[1]/2))            
            if (small_size[0] == 0 or small_size[1] == 0):
                small_size = image.size
            image_small = image_small.resize(small_size, Image.ANTIALIAS)
            
            # run deepDreamRecursive on the scaled down image
            image_small = self.deepDreamRecursive(image_small, layer, iterations, lr, num_downscales-1)
            
            # Scale up the result image to the original size
            image_large = image_small.resize(image.size, Image.ANTIALIAS)
            
            # Blend the two image
            image = ImageChops.blend(image, image_large, BLEND_ALPHA)
        img_result = self.deepDream(image, layer, iterations, lr)
        img_result = img_result.resize(image.size)
        return img_result
    
    def deepDreamProcess(self):
        t0 = time()
        out = self.deepDreamRecursive(self.image, self.layer_id, NUM_ITERATIONS, LR, NUM_DOWNSCALES)
        print('... done in {0:.2f}s'.format(time()-t0))
        return out

# Now, load an image

# In[11]:

if __name__ == '__main__':
    
    
    img = Image.open(IMAGE_PATH)
      
    pyplot.imshow(img)
    pyplot.title("Image loaded from " + IMAGE_PATH)
    
    
    # Perform "deep dream" on it. Note that the computation could take a few minutes.
    
    # In[12]:
    
    _id = np.random.randint(20, 32)
    D = DeepDream(layer_id=_id)
#    D = DeepDream()
    img_deep_dream = D.load(img).deepDreamProcess()
    pyplot.imshow(img_deep_dream)
    pyplot.title("Deep dream image (layer {0})".format(_id))
    
    
    # In[13]:
    
    
    #img_deep_dream.save('deepdream_' + IMAGE_PATH)
    
