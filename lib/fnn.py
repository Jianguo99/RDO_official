"""
@author: jpzxshi
"""
from os import truncate
import torch.nn as nn
from . import initializers  
from . import activations  
class FNN(nn.Module):
    '''Fully connected neural networks.
    '''
    def __init__(self, structure, activation='relu', initializer='default', softmax=False):
        super(FNN,self).__init__()
        self.ind = structure[0]
        self.outd = structure[-1]
        self.layers = len(structure)-1  # 
        self.structure = structure
        
        self.act = activations.get(activation)#nn.ReLU
        self.initializer = initializers.get(initializer)
        self.initializer_zero = initializers.get('zeros')
        self.softmax = softmax
        
        self.net  = self.__init_modules() 
        self.__initialize_parms() 
        
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        net = nn.ModuleList()
        for i in range(self.layers):
            net.append(nn.Linear(self.structure[i],self.structure[i+1]))
            if i+1 != self.layers: 
                net.append(self.act(inplace = True))
        return net

            
    
    def __initialize_parms(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                self.initializer(layer.weight)
                self.initializer_zero(layer.bias)
                # nn.init.constant_(layer.bias, 0)