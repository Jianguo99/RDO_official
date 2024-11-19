"""
@author: jpzxshi
"""
import torch
import torch.nn as nn

from .module import StructureNN
from .fnn import FNN
import numpy as np
from .fno import SpectralConv1d,SpectralConv2d
import torch.nn.functional as F
from einops import rearrange


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        # print(self.weights1[0,0,0])

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)

        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class  FNO1dX_3block(nn.Module):
    def __init__(self, modes, width,output_dim,cellcenters_nx):
        super(FNO1dX_3block, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.output_dim =output_dim
        self.grid = cellcenters_nx
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        # self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.output_dim)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(-1) 
        grid = self.get_grid(x.shape, x.device)
        # print(x.shape,grid.size())
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = self.grid.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class FNO1dX_1block(nn.Module):
    def __init__(self, modes, width,output_dim,cellcenters_nx):
        super(FNO1dX_1block, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.output_dim =output_dim
        self.grid = cellcenters_nx
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        # self.w1 = nn.Conv1d(self.width, self.width, 1)
        # self.w2 = nn.Conv1d(self.width, self.width, 1)
        # self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.output_dim)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(-1) 
        grid = self.get_grid(x.shape, x.device)
        # print(x.shape,grid.size())
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        # print(batchsize, size_x )
        gridx = self.grid.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    


class littleFNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(littleFNN,self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim,output_dim),
                                )

    def forward(self,x):
        return self.net(x)




class FNO2dX(nn.Module):
    def __init__(self, modes1,modes2, width,output_dim,cellcenters_nx):
        super(FNO2dX, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, output_dim)


        self.grid = torch.tensor(cellcenters_nx, dtype=torch.float)

    def forward(self, x):
        x =x.unsqueeze(-1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = self.grid.repeat([batchsize, 1, 1,1])
        return gridx.to(device)



class SelfAttention(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self,x,mask=None):
        assert x.dim() ==3, '3D tensor must be provided'
        qkv = self.to_qvk(x)

        #decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))
        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        return torch.einsum('b i j , b j d -> b i d', attention, v)



        

class BranchNet1d_selfAttentionv1(nn.Module):
    """ BranchNet"""
    def __init__(self, 
                    branchNet,
                    TrunkNet,cellcenters_nx,
                    problem_name,
                    fourierblock,
                    integral_type,
                    integral_interval):
        super( ).__init__()
        self.branchNet = branchNet
        self.grid = torch.tensor(cellcenters_nx, dtype=torch.float).cuda()
        self.TrunkNet =TrunkNet
        self.hidden_size = TrunkNet[-1]
        self.integral_type = integral_type
        self.integral_interval = torch.tensor(integral_interval, requires_grad=False).cuda()
        self.problem_name = problem_name

        self.modus = nn.ModuleDict()
        if self.problem_name in ["DarcyTriangular"]:
            self.modus['FNO'] = FNO1dX_1block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
            self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])   
        else:
            if self.problem_name in ['ODE'] or fourierblock==3:
                self.modus['FNO'] = FNO1dX_3block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
                self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])
                self.modus['Linear'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])
            elif fourierblock==1:
                self.modus['FNO'] = FNO1dX_1block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
                self.modus['selfAttention'] = SelfAttention(self.branchNet[-1])   
            
            
    def forward(self,x):
        if self.problem_name in ['ODE']:
            x = self.modus['FNO'](x)
            x = self.modus['selfAttention'](x)
            x = self.modus['Linear'](x)
            
        if self.problem_name in ["DarcyTriangular"]:
            x = self.modus['FNO'](x)
            x = self.modus['selfAttention'](x)
            
        if self.integral_type == "mean":
            x= torch.mean(x,dim=1)
        elif self.integral_type == "sum":
            x= torch.sum(x,dim=1)* self.integral_interval
        else:
            raise NotImplementedError
        return x
    
    
class BranchNet1d_fno(nn.Module):
    """ BranchNet"""
    def __init__(self, 
                    branchNet,
                    TrunkNet,cellcenters_nx,
                    problem_name,
                    fourierblock):
        super( ).__init__()
        self.branchNet = branchNet
        self.grid = torch.tensor(cellcenters_nx, dtype=torch.float).cuda()
        self.TrunkNet =TrunkNet
        self.hidden_size = TrunkNet[-1]
        self.problem_name = problem_name

        self.modus = nn.ModuleDict()
        if self.problem_name in ['ODE'] or fourierblock==3:

            self.modus['FNO'] = FNO1dX_3block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
            self.modus['Linear1'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])
        elif self.problem_name in ["DarcyTriangular"] or  fourierblock==1:
            self.modus['FNO'] = FNO1dX_1block(self.branchNet[0],self.branchNet[1],self.branchNet[-1],self.grid)
              

    def forward(self,x):
        if self.problem_name in ['ODE']:
            x = self.modus['FNO'](x)
            x = self.modus['Linear1'](x)
            x= torch.mean(x,dim=1)
        if self.problem_name in ["DarcyTriangular"]:
            x = self.modus['FNO'](x)
        return x
    
class BranchNet1d_attention(nn.Module):
    """BranchNet"""
    def __init__(self, 
                    branchNet,
                    TrunkNet,cellcenters_nx,
                    problem_name,
                    fourierblock):
        super( ).__init__()
        self.branchNet = branchNet
        self.grid = torch.tensor(cellcenters_nx, dtype=torch.float).cuda()
        self.TrunkNet =TrunkNet
        self.hidden_size = TrunkNet[-1]
        self.problem_name = problem_name

        self.modus = nn.ModuleDict()
        if self.problem_name in ['ODE'] or fourierblock==3:
            self.modus['Linear1'] = littleFNN(2,self.hidden_size,branchNet[-1])
            self.modus['selfAttention'] = SelfAttention(branchNet[-1])
            self.modus['Linear2'] = littleFNN(branchNet[-1],self.hidden_size,TrunkNet[-1])
        elif self.problem_name in ["DarcyTriangular"] or  fourierblock==1:
            self.modus['selfAttention'] = SelfAttention(branchNet[-1])

                                    
    def forward(self,x):
        x =x.unsqueeze(-1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        if self.problem_name in ['ODE']:
            x = self.modus['Linear1'](x)
            x = self.modus['selfAttention'](x)
            x =  self.modus["Linear2"](x)
            x= torch.mean(x,dim=1)
        if self.problem_name in ["DarcyTriangular"]:
            x = self.modus['selfAttention'](x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = self.grid.repeat([batchsize, 1,1])
        return gridx.to(device)

class BranchNet2d_selfAttentionv1(nn.Module):
    """BranchNet"""
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




class RDO(StructureNN):
    '''
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self,
                input_dim,
                branchNet, 
                TrunkNet,
                cellcenters_nx,
                activation='relu', 
                kernel_initializer='Glorot normal',
                integral_type = "mean",
                integral_interval = 1,
                input_domain_dim=1,
                problem_name = "ODE",
                fourierblock=3,
                hard_constraint=False):
        super().__init__()
        self.input_dim = input_dim
        self.branchNet = branchNet
        self.trunk_dim = TrunkNet[0]
        self.TrunkNet = TrunkNet
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.cellcenters_nx = cellcenters_nx
        self.input_domain_dim = input_domain_dim
        self.problem_name = problem_name
        self.fourierblock = fourierblock
        self.integral_type =  integral_type
        if self.integral_type == "mean":
            self.integral_interval = 1
        else:
            self.integral_interval = integral_interval
        self.hard_constraint = hard_constraint
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.input_dim], x[..., self.input_dim:]
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
            modules['Branch'] = BranchNet1d_selfAttentionv1(self.branchNet,self.TrunkNet,self.cellcenters_nx,self.problem_name,self.fourierblock, self.integral_type, self.integral_interval)
        elif self.input_domain_dim==2:
            modules['Branch'] = BranchNet2d_selfAttentionv1(self.branchNet,self.TrunkNet,self.cellcenters_nx)

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
        if self.hard_constraint!=False:
            return self.hard_constraint(output,y)
        else:
            return output

    def Extracter(self,x):
        """
        x: the input field 
        """
        x_branch = self.modus['Branch'](x)
        self.x_branch = x_branch.squeeze(-1)
