from typing import List, Union

from utils.common import BisectList
from .base import DataSourceBase

class Label(BisectList):
    def __init__(self, name, sources:Union[DataSourceBase, List[DataSourceBase]], id:int=None):
        self.name = name
        self.id = id
        if isinstance(sources, DataSourceBase):
            sources = [sources]
        super().__init__(lst_of_lst=sources, sort_key='name', reset_index=None)
    
    @property
    def sources(self):
        return self.lst_of_lst
    
    def add_sources(self, sources:Union[DataSourceBase, List[DataSourceBase]]):
        if isinstance(sources, DataSourceBase):
            sources = [sources]
        for s in sources:
            if s in self.sources:
                print(f"Source '{s.name}' already exists in label {self.name}")
                sources.remove(s)
            
        self.set_lst_of_lst(sources + self.sources)
        
    def __getitem__(self, index):
        if index < 0: index += len(self)
        assert index < len(self), f"Index {index} out of range: max = {len(self)-1}"
        return self.find_in_next(index)