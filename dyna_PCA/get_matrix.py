import torch
import numpy as np
import pandas as pd

def get_matrix(array):
    n_sim=  array.shape[0]
    n_timesteps = array.shape[1]
    n_nodes = array.shape[3]


    x_slice= array[:,:,0,:] #torch.Size([50, 102, 114])
    y_slice=array[:,:,1,:]
    z_slice= array[:,:,2,:]

    # x_resh= x_slice.permute(1,0,2) #torch.Size([102, 50 ,114])
    # y_resh= y_slice.permute(1,0,2)
    # z_resh = z_slice.permute(1,0,2)

    # x_disp= x_resh.reshape(n_timesteps,-1)
    # y_disp= y_resh.reshape(n_timesteps,-1)
    # z_disp= z_resh.reshape(n_timesteps,-1)

    # matrix = torch.cat((x_disp, y_disp, z_disp), 0) #306,5700
    # matrix_t = matrix.T #5700, 306 *, times

    x_disp_stack2 = torch.reshape((x_slice),(((n_sim*n_timesteps),(n_nodes))))
    y_disp_stack2 = torch.reshape((y_slice),(((n_sim*n_timesteps),(n_nodes))))
    z_disp_stack2 = torch.reshape((z_slice),(((n_sim*n_timesteps),(n_nodes))))

    Matrix2= torch.cat((x_disp_stack2, y_disp_stack2, z_disp_stack2), axis=1) # 5100, 342 ==  n_sim*ts,disp

    return Matrix2
