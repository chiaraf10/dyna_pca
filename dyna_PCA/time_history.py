import torch 

def add_time(fem, t):
    # 'This time adds the time variable in the fem paramter input.

    # given 2 paramters and t
    # the prm vector will result in :

    #         a1,b1,t0,
    #         a1,b1,t1
    #         ...
    #         an,bn,tn
    # inserting time variable in the regression input'
    tmp =[]
    n_par=fem.shape[1]
    for i in range(len(fem)):
        for j in range(len(t)): 
            temp =[fem[i][0],fem[i][1],fem[i][2],fem[i][3],fem[i][4],t[j]] 
            tmp.append(temp)

    prm= torch.tensor(tmp)
    return prm