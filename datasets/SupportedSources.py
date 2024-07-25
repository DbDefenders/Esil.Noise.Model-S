from deprecated import deprecated
from typing import Union
from .models.base import DataSourceBase
from .models.sources import (
    US8KDataSource,
    ESC50DataSource,
    ProvinceDataSource,
    BirdclefDataSource,
)
from utils.common import get_func_params
from copy import deepcopy
from functools import lru_cache

from utils import config
from enum import Enum

__all__ = [
    "SupportedSourceTypes",
    "get_data_source"
]

datasources_info = config.data_sources

_SOURCES_DICT:dict = {
    'US8K': US8KDataSource,
    'ESC50': ESC50DataSource,
    'Birdclef': BirdclefDataSource,
    'TRAFFIC': ProvinceDataSource,
    'NATURE': ProvinceDataSource,
    'INDUSTRIAL': ProvinceDataSource,
    'SOCIAL': ProvinceDataSource,
    'CONSTRUCTIONAL': ProvinceDataSource,
}

_KWARGS_DICT:dict = {
    'US8K': datasources_info["US8K"],
    'ESC50': datasources_info["ESC50"],
    'Birdclef': datasources_info["Birdclef"],
    'TRAFFIC': {**datasources_info["Province"], "name":"交通噪声", "name_en": 'Traffic_noise'},
    'NATURE': {**datasources_info["Province"], "name":"自然噪声", "name_en": 'Natural_noise'},
    'INDUSTRIAL': {**datasources_info["Province"], "name":"工业噪声", 'name_en': 'Industrial_noise'},
    'SOCIAL': {**datasources_info["Province"], "name":"社会噪声", "name_en": "Social_noise"},
    'CONSTRUCTIONAL': {**datasources_info["Province"], "name":"建筑施工噪声", "name_en": 'Construction_noise'},
}


def get_class_and_args(type_name:str,  **kwargs)->dict:
    class_ = _SOURCES_DICT.get(type_name, None)
    args:dict = deepcopy(_KWARGS_DICT.get(type_name, None))
    args.update(kwargs)
    
    if class_ is None:
        raise ValueError(f"Unsupported source type: {type_name}")
    
    if class_ == ProvinceDataSource:
        '''
        如果是ProvinceDataSource，判断是否是使用英文文件夹
        '''
        if args.get('use_en'):
            args['name'] = args['name_en']
            
        if 'name_en' in args.keys():
            args.pop('name_en')
            
    return {
        'class': class_,
        'args': args
    }
    
class SupportedSourceTypes(Enum):
    '''
    - 查看指定数据源的构建函数的参数：SupportedSourceTypes.@TYPE.value or SupportedSourceTypes.@TYPE.get_params()
    - 获取指定数据源的类：SupportedSourceTypes.@TYPE.get_class()
    - 获取指定数据源的初始化参数：SupportedSourceTypes.@TYPE.get_kwargs()
    - 获取指定数据源的实例：SupportedSourceTypes.@TYPE.get()
    '''
    US8K = get_func_params(US8KDataSource.__init__)
    ESC50 = get_func_params(ESC50DataSource.__init__)
    BIRDCLEF = get_func_params(BirdclefDataSource.__init__)
    TRAFFIC = get_func_params(ProvinceDataSource.__init__, type_='TRAFFIC')
    NATURE = get_func_params(ProvinceDataSource.__init__, type_='NATURE')
    INDUSTRIAL = get_func_params(ProvinceDataSource.__init__, type_='INDUSTRIAL')
    SOCIAL = get_func_params(ProvinceDataSource.__init__, type_='SOCIAL')
    CONSTRUCTIONAL = get_func_params(ProvinceDataSource.__init__, type_='CONSTRUCTIONAL')
    
    def get(self, **kwargs):
        '''
        get_data_source的替代版本，用于获取指定数据源的实例
        :param kwargs: 数据源初始化参数
        :return: 数据源实例
        '''
        data_source_class = self.get_class()
        data_source_args = self.get_kwargs(**kwargs)
        return data_source_class(**data_source_args)
    
    def get_params(self)->dict:
        return self.value
    
    def get_class(self, type_name:str=None):
        '''
        获取指定数据源的类
        '''
        if type_name is None:
            type_name = self.name
        return _SOURCES_DICT.get(type_name, None)
    
    def get_kwargs(self, type_name:str=None, **kwargs)->dict:
        '''
        获取指定数据源的初始化参数
        '''
        
        if type_name is None:
            type_name = self.name
            
        value = get_class_and_args(type_name, **kwargs)
        return value['args']
    
@deprecated(reason='use SupportedSourceTypes.@TYPE.get() instead')
@lru_cache(maxsize=len(SupportedSourceTypes))
def get_data_source(source_type: Union[str,SupportedSourceTypes])->DataSourceBase:
    '''
    按需获取指定数据源的实例
    
    :param source_type: 数据源类型
    :return: 数据源实例
    '''
    if isinstance(source_type, str):
        source_type = SupportedSourceTypes[source_type.upper()]
        
    value = get_class_and_args(source_type.name)
    
    return value["class"](**value["args"])
