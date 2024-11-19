import numpy as np
from torch.utils import data
import os
import joblib
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import time
import argparse
import json
import sys
from lib.rdo import *
from lib.deeponet import *
from lib.fno import FNO1dX
from lib.utils import *


MODEL_SAVE_ROOT = "pth"
def args_to_dict(args):
    return vars(args)

def save_dict_to_file(args_dict, filename='args.json'):
    with open(filename, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)   

class FNODataset(data.Dataset):
    def __init__(self,data,r) -> None:
        super().__init__()
        self.coeff = data['coeff'][:,::r]
        numexamples,size_x = self.coeff.shape
        self.resolution = size_x
        self.sol = data['sol'][:,::r]

        
    def __getitem__(self,index):
        return self.coeff[index,:].reshape(-1,1),self.sol[index,:].reshape(-1,1)

    def __len__(self,):
        return self.sol.shape[0]

class ComplexGeometryDataset(data.Dataset):
        def __init__(self, data,r, coordinates_output) -> None:
            super().__init__()
            
            self.coeff = data['coeff'][:,::r]
            numexamples,size_x = self.coeff.shape
            trainDataU = data['sol'][:,::r]
            left_boundary = np.ones((numexamples,1))
            right_boundary = np.zeros((numexamples,1))
            sol = np.concatenate([left_boundary,trainDataU,right_boundary],axis=1)    
            self.resolution = sol.shape[1]
            self.sol = sol.reshape(-1)

            # print(self.sol.shape)
            self.cellcenters = np.concatenate([np.zeros((1,1)),coordinates_output[::r],np.ones((1,1))])

        def __getitem__(self,index):
            return np.concatenate([self.coeff[index//self.resolution,:],self.cellcenters[index%self.resolution].reshape(-1)]), self.sol[index]

        def __len__(self,):
            return self.sol.shape[0]
        
        
        
def main(args):
    model = args.model
    lx = args.lx
    r = args.r
    seed = args.seed
    fix_random_seed(seed)
    print(args)
    ##################
    # Hyper-parameters for training
    ####################
    
    
    BATCH_SIZE=1024
    EPOCHS = 1000
    step_size = 200
    LR = 0.001
        
    #####################
    # Loading datasets
    #####################
    data_path1 = f"data/lx={lx}/TrainData.pkl"
    data_path2 = f"data/lx={lx}/ValData.pkl"
    coordinates_inputfield = joblib.load(f"data/lx={lx}/coordinates_inputfield.pkl")
    coordinates_output = joblib.load(f"data/lx={lx}/coordinates_output.pkl")
    training_data =  joblib.load(data_path1)
    validation_data =  joblib.load(data_path2)
    coordinates_inputfield_dim = coordinates_inputfield.shape[0]
    inputfield_resolution = coordinates_inputfield_dim//r +coordinates_inputfield_dim%r
    integral_interval = 1/ (inputfield_resolution-1)
    print("inputfield_resolution:",inputfield_resolution)
    save_model_name = f"model={model}_input_resolution={str(inputfield_resolution)}_seed={seed}"
    
    if model == "RDO" or model == "DeepONet" or model == "DeepONet_fno_transformer":
        train_dataset  = ComplexGeometryDataset(training_data,r, coordinates_output)
        val_dataset  = ComplexGeometryDataset(validation_data,r, coordinates_output)
        train_dataloader  = data.DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=False)
        val_dataloader  = data.DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=False)
        
        
        if "RDO" in model:
            integral_type = args.integral_type
            if integral_type == "sum":
                save_model_name += f"_interval={integral_interval}"

            modes=16
            FNOWidth = 32
            FNOOutputDim = 64
            if model == "RDO":
                MyModel = RDO(input_dim=inputfield_resolution,
                            branchNet= [modes,FNOWidth,FNOOutputDim],  
                            TrunkNet =  [1,100,100,100],
                            cellcenters_nx  = coordinates_inputfield[::r],
                            kernel_initializer="Glorot normal",
                            integral_type = integral_type,
                            integral_interval = integral_interval,
                            )       
            else:
                raise NotImplementedError
              

        else:
            if model == "DeepONet":
                
                MyModel = DeepONet(branchNet= [inputfield_resolution,100, 100,100],
                        TrunkNet =  [1,100,100,100],
                        kernel_initializer="Glorot normal")
            elif model == "DeepONet_fno_transformer":
                modes=16
                FNOWidth = 32
                FNOOutputDim = 64
                MyModel = DeepONet_fno_transformer(branchNet= [inputfield_resolution, modes,FNOWidth,FNOOutputDim], 
                                                    TrunkNet =  [1,100,100,100],
                                                    cellcenters_nx  = coordinates_inputfield[::r],
                                                    kernel_initializer="Glorot normal")
            else:
                raise NotImplementedError
            
    elif model == "FNO":
        modes=16
        width=128
        train_dataset  = FNODataset( training_data,r)
        val_dataset  = FNODataset( validation_data,r)
        train_dataloader  = data.DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=False)
        val_dataloader  = data.DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=False)
        MyModel = FNO1dX(modes,width,torch.from_numpy(np.array(coordinates_inputfield[::r],dtype=np.float32)).cuda())

    save_model_name += ".pth"
    if not os.path.exists(f"{MODEL_SAVE_ROOT}/lx={lx}"):
        os.makedirs(f"{MODEL_SAVE_ROOT}/lx={lx}")
    save_model_path = os.path.join(f"{MODEL_SAVE_ROOT}/lx={lx}",save_model_name) 

    MyModel.cuda()
    print("The number of model parameters is",dnn_paras_count(MyModel))  
    optimizer =  torch.optim.Adam(MyModel.parameters(),lr= LR,weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5,verbose=False)    
    criterion = F.mse_loss  

    loss_best = 999
    training_start_time = time.time()
    with tqdm(total=EPOCHS, desc = "Epoch") as _tqdm:
        for epoch in range(1,EPOCHS+1):
            avg_loss =0
            MyModel.train()
            for j ,(input,label) in enumerate(train_dataloader):
                input =  torch.from_numpy(np.array(input,dtype=np.float32)).cuda()
                label =  torch.from_numpy(np.array(label,dtype=np.float32)).cuda()
                optimizer.zero_grad()  
                pred = MyModel(input)   
                loss = criterion(pred,label) 
                loss.backward()   

                optimizer.step()   
                avg_loss = (avg_loss*np.maximum(0,j) + loss.data.cpu().numpy())/(j+1)
                _tqdm.set_postfix({  "loss": '{:.6f}'.format(avg_loss)})
                
            scheduler.step()
            MyModel.eval()
            val_avg_loss = 0
            for j ,(input,label) in enumerate(val_dataloader):
                input =  torch.from_numpy(np.array(input,dtype=np.float32)).cuda()
                label =  torch.from_numpy(np.array(label,dtype=np.float32)).cuda()
                optimizer.zero_grad()  
                pred = MyModel(input)   
                loss = criterion(pred,label) 
                val_avg_loss = (val_avg_loss*np.maximum(0,j) + loss.data.cpu().numpy())/(j+1)
            if val_avg_loss < loss_best:
                torch.save(MyModel.state_dict(), save_model_path) 
                loss_best =  val_avg_loss
                best_epoch = epoch
                best_train_loss = avg_loss
                best_val_loss = val_avg_loss
            _tqdm.update(1)
    print("Best model at step {:d}:".format(best_epoch))
    print("  train loss: {:.8e}".format(best_train_loss))
    print("  validation loss: {:.8}".format(best_val_loss))
    training_time = time.time()-training_start_time
    print("  Total training time: {:.8}".format(training_time))
    
    #####################
    # Saving the final results
    #####################
    args_dict = args_to_dict(args)
    args_dict["best_model_step"] = best_epoch
    args_dict["best_train_loss"] = best_epoch
    args_dict["best_val_loss"] = best_val_loss
    args_dict["training_time"] = training_time
    file_without_extension = os.path.splitext(save_model_path)[0]

    save_dict_to_file(args_dict, file_without_extension+".json")

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates conformal predictors',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lx', type=float, default=0.1, help='the length correlation')
    parser.add_argument('--model', type=str, default='RDO', help='model')
    parser.add_argument('--r', type=int, default=4, help='the resolution of input field')
    parser.add_argument('--integral_type', type=str, default='mean', help='model')
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--gpu', type=int, default=3, help='chose gpu id')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)
