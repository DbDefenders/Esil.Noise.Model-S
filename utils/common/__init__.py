import json
import numpy as np
from typing import List, Union
from bisect import bisect_left
from functools import singledispatch

def create_repr_str(obj, properties:list):
    """
    Returns a string representation of an object with specified properties.
    """
    info_str = []
    for i in properties:
        if i is not None:
            val = getattr(obj, i)
            if val is None:
                continue
            elif isinstance(val, list) and len(val) == 0:
                continue
            elif isinstance(val, str):
                info_str.append(f'{i}="{val}"')
            else:
                info_str.append(f"{i}={val}")
    info_str = ", ".join(info_str)
    return f'{type(obj).__name__}({info_str})'

@singledispatch
def get_child(index, childs:List,key:str='name'):
    raise TypeError("Unsupported index type: {}".format(type(index)))

@get_child.register
def _(index: int, childs, key:str='name'):
    assert index < len(childs), f"Index out of range: {len(childs)}"
    return childs[index]

@get_child.register
def _(index: str, childs, key:str='name'):
    for child in childs:
        if getattr(child, key) == index:
            return child
    raise ValueError(f"Child with '{key}' '{index}' not found")

class BisectList:
    def __init__(self, lst_of_lst:List[List], primary_key:str, reset_index:bool, length:int=None):
        '''
        lst_of_lst: a list of lists to be bisected.
        length: the total length of the bisected list.
        '''
        
        self.by = primary_key
        if len(lst_of_lst) > 0:
            assert hasattr(lst_of_lst[0], primary_key), f"Primary key '{primary_key}' not found in {type(lst_of_lst[0])} members."
        self.reset_index = reset_index
        self.length = length
        self.set_lst_of_lst(lst_of_lst)
        
    def set_lst_of_lst(self, lst_of_lst:List[List]):
        self.lst_of_lst = lst_of_lst
        self.order()
        
    def order(self, *, by:str=None):
        if by is None:
            by = self.by
        self.lst_of_lst.sort(key=lambda x: getattr(x, by))
        self.bisert_breakpoints = []
        start = 0
        for index, child in enumerate(self.lst_of_lst):
            if self.reset_index: setattr(child, by, index)
            start += len(child)
            self.bisert_breakpoints.append(start)
        self.left_bps = [0] + self.bisert_breakpoints[:-1]
        self.right_bps = list(map(lambda x: x-0.5, self.bisert_breakpoints))
        
    def find_in_next(self, index)->tuple:
        next_idx = bisect_left(self.right_bps, index)
        new_idx = index - self.left_bps[next_idx]
        return self.lst_of_lst[next_idx][new_idx]
    
    def has_next(self):
        if len(self.lst_of_lst) > 0:
            return True
        else:
            return False
        
    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        elif self.lst_of_lst:
            return sum(len(c) for c in self.lst_of_lst)
        else:
            print("Unknown length.")
            return 0
        

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
        
def save_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)
    return file_path