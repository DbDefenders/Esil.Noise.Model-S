import torch

def tensor_to_number(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            try:
                result = result.item()
            except ValueError:
                pass
        return result
    return wrapper