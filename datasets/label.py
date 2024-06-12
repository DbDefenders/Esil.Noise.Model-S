from typing import List, Union

from utils.common import BisectList
from .base import DataSourceBase

class Label(BisectList):
    def __init__(self, name, sources:Union[DataSourceBase, List[DataSourceBase]], id:int=None):
        self.name = name
        self.id = id
        
        super().__init__(lst_of_lst=sources, primary_key='name', reset_index=False)
    
    @property
    def sources(self):
        return self.lst_of_lst
    
    def add_source(self, source:DataSourceBase):
        self.set_lst_of_lst([source] + self.sources)
        
    def __getitem__(self, index):
        if index < 0: index += len(self)
        assert index < len(self), f"Index {index} out of range: max = {len(self)-1}"
        return self.find_in_next(index)