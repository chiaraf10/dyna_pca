import torch
import numpy as np

#normalize the dataset by dividing the max nicrement netween two timesteps

def _max_(sim):   
    array= sim.permute(0,3,1,2)[:,:,:,:] #sim,node,time,xyz
    max_x, max_y,max_z=torch.zeros(3)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for t in range(array.shape[2]-1):
                incr_x,incr_y,incr_z = array[i,j,t+1,:]  - array[i,j,t,:]
                if incr_x > max_x:
                    max_x = incr_x
                if incr_y > max_y:
                    max_y = incr_y
                if incr_z > max_z:
                    max_z = incr_z
            
    #print(max_x, max_y,max_z)
    max_ = np.max([max_x, max_y,max_z])
    sim = sim / max_
    return sim,max_
