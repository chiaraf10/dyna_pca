import numpy as np
import pandas as pd


def get_matrix(array):
    n_sim = array.shape[0]  # from [n_simulation, n_timesteps, xyz, n_nodes]
    n_timesteps = array.shape[1]
    n_nodes = array.shape[3]

    x_slice = array[:, :, 0, :]  # [n_simulations, timesteps, n_nodes]
    y_slice = array[:, :, 1, :]
    z_slice = array[:, :, 2, :]

    x_disp_stack2 = np.reshape((x_slice), (((n_sim * n_timesteps), (n_nodes))))
    y_disp_stack2 = np.reshape((y_slice), (((n_sim * n_timesteps), (n_nodes))))
    z_disp_stack2 = np.reshape((z_slice), (((n_sim * n_timesteps), (n_nodes))))

    pca_matrix = np.concatenate(
        (x_disp_stack2, y_disp_stack2, z_disp_stack2), axis=1
    )  # n_sim*ts,disp

    return pca_matrix
