import torch
def MSE(input,target): 
    # Mean Square error function. The mean squared error (squared L2 norm) between each element in the input xx and target yy.
    # it can be either sum or mean. Pyotrch default version uses the mean.
    # MSE is calculated by taking the difference between `y` and our prediction, then square those values. 
    # We take these new numbers (square them), add all of that together to get a final value, finally divide this number by y again. 
    #  1/N * mean(y_true - y_target)^2. IT can be iether sum() or mean()

    
    loss = torch.mean((input - target) ** 2)  
    # %%
    ### TEST
    # torch.manual_seed(42)

    # input = torch.randn(3, 5, requires_grad=True, )
    # target = torch.randn(3, 5)

    # loss = MSE(input,target)

    # print(loss) == tensor(1.0192, grad_fn=<MeanBackward0>)
    # If computed with the MSE from pytorch ==tensor(1.0192, grad_fn=<MseLossBackward0>)

    # %%
    return loss