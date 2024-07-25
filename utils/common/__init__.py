import json
import numpy as np
from typing import List
from typing import Callable, Dict, Any
from abc import ABC, abstractmethod
from bisect import bisect_left
from functools import singledispatch
from IPython.display import clear_output
import inspect
import time

__all__ = ["clear_jupyter", "create_repr_str", "get_child", "BisectList", "save_json"]

def get_func_params(func: Callable) -> Dict[str, Any]:
    parameters = inspect.signature(func).parameters
    ret = {
        "Function": func.__name__,
        "Parameters": [p.name for p in parameters.values()],
        "ID": f'{int(time.time())}-{id(func)}'
    }
    return ret

# 示例函数
def example_function(a, b, c=10, *args, **kwargs):
    pass

# 调用 get_func_params 函数
params_info = get_func_params(example_function)
print(params_info)



def clear_jupyter():
    try:
        clear_output()
    finally:
        pass


def create_repr_str(obj, properties: list):
    """
    Returns a string representation of an object with specified properties.
    """
    info_str = []
    for i in properties:
        if not hasattr(obj, i):
            continue
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
    return f"{type(obj).__name__}({info_str})"


@singledispatch
def get_child(index, childs: List, key: str = "name"):
    raise TypeError("Unsupported index type: {}".format(type(index)))


@get_child.register
def _(index: int, childs, key: str = "name"):
    assert index < len(childs), f"Index out of range: {len(childs)}"
    return childs[index]


@get_child.register
def _(index: str, childs, key: str = "name"):
    for child in childs:
        if getattr(child, key) == index:
            return child
    raise ValueError(f"Child with '{key}' '{index}' not found")


class BisectList(ABC):
    def __init__(
        self,
        lst_of_lst: List[List],
        sort_key: str,
        reset_index: str = None,
        length: int = None,
    ):
        """
        lst_of_lst: a list of lists to be bisected.
        length: the total length of the bisected list.
        sort_key: the key to sort the list of lists by.
        reset_index: the key to reset the index of each child in the list of lists.
        """

        self._order_by = sort_key
        if len(lst_of_lst) > 0:
            assert hasattr(
                lst_of_lst[0], sort_key
            ), f"Primary key '{sort_key}' not found in {type(lst_of_lst[0])} members."
        self._reset_index = reset_index
        self.length = length
        self.set_lst_of_lst(lst_of_lst)

    def set_lst_of_lst(self, new_val: List[List]):
        try:
            self._lst_of_lst = list(set(new_val))
        except Exception as e:
            print(f"Error in set_lst_of_lst: {e}")
            raise e
        self.__order()

    def __order(self, *, by: str = None):
        if by is None:
            by = self._order_by
        else:
            self._order_by = by
        self._lst_of_lst.sort(key=lambda x: getattr(x, by))
        self._bisect_breakpoints = []
        start = 0
        for index, child in enumerate(self._lst_of_lst):
            if isinstance(self._reset_index, str) and hasattr(child, self._reset_index):
                setattr(child, self._reset_index, index)
            start += len(child)
            self._bisect_breakpoints.append(start)
        self._left_bps = [0] + self._bisect_breakpoints[:-1]
        self._right_bps = list(map(lambda x: x - 0.5, self._bisect_breakpoints))

    def get_next_node_idx(self, index: int) -> tuple:
        next_idx = bisect_left(self._right_bps, index)
        new_idx = index - self._left_bps[next_idx]
        return next_idx, new_idx

    def find_in_next_node(self, index) -> tuple:
        next_idx, new_idx = self.get_next_node_idx(index)
        return self._lst_of_lst[next_idx][new_idx]

    def has_next(self):
        if len(self._lst_of_lst) > 0:
            return True
        else:
            return False

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        elif self._lst_of_lst:
            return sum(len(c) for c in self._lst_of_lst)
        else:
            print("Unknown length.")
            return 0

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ not implemented")


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
    with open(file_path, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)
    return file_path
