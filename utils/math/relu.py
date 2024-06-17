import torch

def limit_relu(x:torch.Tensor, limit=10, rep=0):
    '''小于百分之limit分位的数值，都用rep替换（默认x的值在0到1之间）'''
    assert limit>=0 and limit<=100
    return torch.where(x < limit/100, torch.tensor([rep]), x)

def percent_relu(x:torch.Tensor, percent=10):
    '''小于百分之percent分位的数值，都用该百分位数替换'''
    assert percent>=0 and percent<=100
    k = int((percent / 100) * x.numel())
    per = torch.kthvalue(x.flatten(), k).values
    return torch.where(x < per, torch.tensor([per]), x)