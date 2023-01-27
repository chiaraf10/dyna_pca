import numpy as np 
from sklearn.model_selection import train_test_split
# def divide_shuffle(array,perc,shuffle,random_seed):

#     data_len= len(list(range((array.shape[0]))))
#     print("Number of simulation: \t", data_len)
    
#     data_indices = list(range(data_len))
#     if shuffle:
#         np.random.seed(random_seed)
#         np.random.shuffle(data_indices)
#     print("Shuffled indices:  " , data_indices)

#     data_split = int(np.floor(perc * data_len)) 
#     print("\nTrain size ", data_split)
#     print("Validation size: ",(data_len - data_split))
#     train_idx = data_indices[:data_split]
#     valid_idx = data_indices[data_split:]#
#     #print("\nTrain idx: ", train_idx)
#     train = array[:data_split,:,:,:]
#     valid = array[data_split:,:,:,:]
#     return train,train_idx, valid, valid_idx

def divide_shuffle(array,parameters, test_size,shuffle,random_seed):

    data_len= len(list(range((array.shape[0]))))
    print("Number of simulation: \t", data_len)
    
    #data_indices = list(range(data_len))
    # if shuffle:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(data_indices)
    # print("Shuffled indices:  " , data_indices)

    # data_split = int(np.floor(perc * data_len)) 
    train_idx, valid_idx = train_test_split((list(range(data_len))), test_size=test_size, shuffle=shuffle, random_state=random_seed)
    print("Train size ", len(train_idx))
    print("Validation size: ",len(valid_idx))
    print("TRAIN indexes:", train_idx, "Validation:", valid_idx)
    train = array[train_idx]
    valid = array[valid_idx]

    fem_train= parameters[train_idx]
    fem_valid= parameters[valid_idx]
    return train_idx, valid_idx,train,valid,fem_train,fem_valid


def divide_shuffle_fem(array,perc,shuffle,random_seed):

    data_len= len(list(range((array.shape[0]))))
    #print("Number of simulation: \t", data_len)
    
    data_indices = list(range(data_len))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(data_indices)
    print("Ls dyna parameters shuffled indices:  " , data_indices)

    data_split = int(np.floor(perc * data_len)) 
    # print("\nTrain size ", data_split)
    # print("Validation size: ",(data_len - data_split))
    train_idx = data_indices[:data_split]#
    #print("\nTrain idx: ", train_idx,data_split)
    train = array[:data_split,:]
    valid = array[data_split:,:]
    return train,train_idx, valid