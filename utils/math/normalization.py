import torch

def min_max_normalize(tensor:torch.Tensor):
    '''将数据缩放到0到1之间'''
    return (tensor - tensor.min())/(tensor.max()-tensor.min())

def z_scroe_normailze(tensor:torch.Tensor):
    '''将数据缩放到均值为0，标准差为1的范围内'''
    return (tensor - tensor.mean())/tensor.std()