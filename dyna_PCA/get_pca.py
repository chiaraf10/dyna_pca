
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np
import pandas as pd

def get_pc(train_idx, Matrix,n_Pca,n_timesteps):
    # Get standardize data
    scaler = StandardScaler(with_mean=True, with_std=True) 
    scaler_std = scaler.fit(Matrix)
    y_Data_std = scaler_std.transform(Matrix)

    var = PCA().fit(y_Data_std).explained_variance_
    cumulative_variance = np.cumsum(PCA().fit(y_Data_std).explained_variance_ratio_)

    pca = PCA(n_components = n_Pca)
    y_PCA = pca.fit_transform(pd.DataFrame(y_Data_std))
    y_PCA = scaler_std.inverse_transform(y_PCA, copy=None)
    n_comps = pca.n_components_
    y_PCA_inverse = pca.inverse_transform(y_PCA) # Inverse transfrom of sample data as per PCs to original form
    y_PCA_InvTrans_Data = scaler_std.inverse_transform(y_PCA_inverse, copy=None) # inverse standardization
    pca_mse = mean_squared_error(Matrix,y_PCA_InvTrans_Data)
    pca_mae = mean_absolute_error(Matrix,y_PCA_InvTrans_Data)
    print("The weights of {} PCs: ".format(var[0:n_comps]))
    print("MSE: {} for n_comps: {}".format(pca_mse, n_comps))
    print("MAE: {} for n_comps: {}".format(pca_mae, n_comps))

    ## Flattening PCA scores according to n_samples. For training
    reshape_y_pca2 =[]
    pd.DataFrame(reshape_y_pca2)
    for i in range(0,len(train_idx)):
    #         temp_1 = y_PCA[(i*31+i):(i*31+i+32)].T
        temp_1 = y_PCA[(i*(n_timesteps-1)+i):(i*(n_timesteps-1)+i+n_timesteps)].T
        temp_2 = temp_1.flatten()
        temp = pd.DataFrame(temp_2)
        reshape_y_pca2.append(temp.T)


    np.row_stack(reshape_y_pca2)   
    y_PCA_ReSh = pd.DataFrame(np.row_stack(reshape_y_pca2), index = train_idx)     

    return y_PCA ,y_PCA_ReSh,n_comps, scaler, pca, y_PCA_InvTrans_Data, cumulative_variance