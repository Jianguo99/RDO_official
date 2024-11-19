"""
@author: jpzxshi
"""
import torch
import torch.nn as nn

from .module import StructureNN
from .fnn import FNN
import numpy as np
from einops import rearrange

from .rdo import *
import torch.nn.functional as F


class BranchNet1d_FNOAtt(nn.Module):
    """ BranchNet"""
    def __init__(self, 
                    branchNet,
                    TrunkNet,
                    cellcenters_nx,
                    problem_name,
                    fourierblock):
        super( ).__init__()
        input_dim = branchNet[0]
        self.branchNet = branchNet[1:]
        self.grid = torch.tensor(cellcenters_nx, dtype=torch.float).cuda()
        self.TrunkNet =TrunkNet
        self.hidden_size = TrunkNet[-1]
        self.problem_name = problem_name

        self.modus = nn.ModuleDict()
        if self.problem_name in ["DarcyTriangular"]:
            self.modus['FNO'] = FNO1dX_1block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
            self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])
            self.modus['Linear1'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])
        else:
            if self.problem_name in ['ODE'] or fourierblock==3:

                self.modus['FNO'] = FNO1dX_3block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
                self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])
                self.modus['Linear1'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])
                
            elif fourierblock==1:
                self.modus['FNO'] = FNO1dX_1block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
                self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])
                self.modus['Linear1'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])
            
        self.modus['Linear2'] = littleFNN(input_dim,128,1)



    def forward(self,x):
        x = self.modus['FNO'](x)
        x = self.modus['selfAttention'](x)
        
        x =  self.modus["Linear1"](x)
        x = x.permute(0, 2, 1)
        x = self.modus['Linear2'](x)
        return x
    
class BranchNet2d_FNOAtt(nn.Module):
    """ BranchNet"""
    def __init__(self, branchNet,TrunkNet,cellcenters_nx):
        super().__init__()
        self.branchNet = branchNet
        self.cellcenters_nx = cellcenters_nx
        self.hidden_size = 32   
        self.numLayers = 1
        self.h0 = torch.zeros(self.numLayers,self.hidden_size,self.hidden_size) 
        self.modus = nn.ModuleDict()
        self.modus['FNO'] = FNO2dX(self.branchNet[0],self.branchNet[1],self.branchNet[2],self.branchNet[3],self.cellcenters_nx)
        self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])
        self.modus['Linear'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])               

    def forward(self,x):
        x = self.modus['FNO'](x)
        x = torch.flatten(x,1,2)
        x = self.modus['selfAttention'](x)
        x = self.modus['Linear'](x)
        x= torch.mean(x,dim=1)
        return x



class DeepONet_fno_transformer(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self,  
                branchNet, 
                TrunkNet,
                cellcenters_nx,
                activation='relu', 
                kernel_initializer='Glorot normal',
                input_domain_dim=1,
                problem_name = "ODE",
                fourierblock=3,
                hard_constraint=False):
        super(DeepONet_fno_transformer, self).__init__()
        
        self.branch_dim = branchNet[0]
        self.branchNet = branchNet
        self.trunk_dim = TrunkNet[0]
        self.TrunkNet = TrunkNet
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.cellcenters_nx = cellcenters_nx
        self.input_domain_dim = input_domain_dim
        self.problem_name = problem_name
        self.fourierblock = fourierblock
        self.hard_constraint = hard_constraint
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., self.branch_dim:]

        x_branch = self.modus['Branch'](x_branch)
        x_branch = x_branch.squeeze(-1)
        x_trunk = self.modus['Trunk'](x_trunk)
        x_trunk = torch.relu(x_trunk) 
        output = (torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']).squeeze()
        if self.hard_constraint!=False:
            return self.hard_constraint(output,x[1])
        else:
            return output
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.input_domain_dim==1:
            modules['Branch'] = BranchNet1d_FNOAtt(self.branchNet,self.TrunkNet,self.cellcenters_nx,self.problem_name,self.fourierblock)
        elif self.input_domain_dim==2:
            modules['Branch'] = BranchNet1d_FNOAtt(self.branchNet,self.TrunkNet,self.cellcenters_nx)

        modules['Trunk'] = FNN(self.TrunkNet,self.activation, self.kernel_initializer)
        return modules
            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        return params
    
    def predict(self,y):
        """
        y: the corrdinate of output point
        """
        x_trunk = self.modus['Trunk'](y)
        x_trunk = torch.relu(x_trunk) 
        # (batchsize,width) (batchsize,points,width)
        res= torch.einsum("bw,bpw->bp", self.x_branch, x_trunk)
        # return (torch.sum(self.x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']).squeeze()
        output = res + self.params['bias']
 
        return output

    def Extracter(self,x):
        """
        x: the input field 
        """
        x_branch = self.modus['Branch'](x)
        self.x_branch = x_branch.squeeze(-1)
    
    
class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self, branchNet, TrunkNet,
                 activation='relu', kernel_initializer='Glorot normal'):
        super(DeepONet, self).__init__()
        self.branch_dim = branchNet[0]
        self.branchNet = branchNet
        self.trunk_dim = TrunkNet[0]
        self.TrunkNet = TrunkNet
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., self.branch_dim:]
        x_branch = self.modus['Branch'](x_branch)
        x_trunk = self.modus['Trunk'](x_trunk)
        x_trunk = torch.relu(x_trunk) 
        # nn.ReLU(x_trunk,inplace=True)
        
        return (torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']).squeeze()
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['Branch'] = FNN(self.branchNet,self.activation, self.kernel_initializer)
        modules['Trunk'] = FNN(self.TrunkNet,self.activation, self.kernel_initializer)
        return modules
            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        return params
    
    def predict(self,y):
        """
        y: the corrdinate of output point
        """
        x_trunk = self.modus['Trunk'](y)
        x_trunk = torch.relu(x_trunk) 
        # (batchsize,width) (batchsize,points,width)
        res= torch.einsum("bw,bpw->bp", self.x_branch, x_trunk)
        # return (torch.sum(self.x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']).squeeze()
        output = res + self.params['bias']
 
        return output

    def Extracter(self,x):
        """
        x: the input field 
        """
        x_branch = self.modus['Branch'](x)
        self.x_branch = x_branch.squeeze(-1)



class Conv(nn.Module):
    def __init__(self,output_dim) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,5,2),
            nn.ReLU(),
            nn.Conv2d(64,128,5,2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(12800, 128),
            # nn.Linear(3200, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    
    def forward(self,x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

        
        

class DeepONet2D(StructureNN):
    '''Deep operator network.
    Input: this is 2-dim function
    Output: [batch size, 1]
    '''
    def __init__(self, TrunkNet,
                 activation='relu', kernel_initializer='Glorot normal'):
        super(DeepONet2D, self).__init__()
        self.trunk_dim = TrunkNet[0]
        self.TrunkNet = TrunkNet
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        
    def forward(self, x):
        x_branch, x_trunk = x[0], x[1]
        x_branch = torch.unsqueeze(x_branch,dim=1)
        # print(x_branch.shape)
        x_branch = self.modus['Branch'](x_branch)
        x_trunk = self.modus['Trunk'](x_trunk)
        x_trunk = torch.relu(x_trunk) 
        # nn.ReLU(x_trunk,inplace=True)
        
        return (torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']).squeeze()
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['Branch'] = Conv(self.TrunkNet[-1])
        modules['Trunk'] = FNN(self.TrunkNet,self.activation, self.kernel_initializer)
        return modules
            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        return params