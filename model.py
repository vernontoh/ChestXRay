

import torch
from torch import nn

class ConvolutionBlock(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(16,16,kernel_size=(3,3),padding='same')
        self.batchnorm = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=(2,2))
        
        
       
    def forward(self,inp):
        
        x = self.conv(inp)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = x+inp ## Skip connection for each block
        x = self.pooling(x)
        
        return x  
        
class NaiveConvolutionNetwork(nn.Module):
    
    def __init__(self,n_blocks=6):
        super().__init__()
        self.blocks = [nn.Conv2d(3,16,kernel_size=(3,3))]+[ConvolutionBlock() for _ in range(n_blocks)]
        self.layers = nn.Sequential(*self.blocks)
        self.pooling = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(144,28)
        self.fc2 = nn.Linear(28,14)
        
    def forward(self,x):
        
        x = self.layers(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.fc(x))
        x = self.fc2(x)
        
        return x
        
        
    


 
