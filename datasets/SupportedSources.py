from .models.base import DataSourceBase
from .models.sources import (
    US8KDataSource,
    ESC50DataSource,
    ProvinceDataSource,
    BirdclefDataSource,
)
from functools import lru_cache

from utils import config
from enum import Enum

__all__ = [
    "SupportedSourceTypes",
    "get_data_source"
]

datasources_info = config.data_sources

class SupportedSourceTypes(Enum):
    US8K = {"class": US8KDataSource, "args":{**datasources_info["US8K"]}}
    ESC50 = {"class": ESC50DataSource, "args":{**datasources_info["ESC50"]}}
    BIRDCLEF = {"class": BirdclefDataSource, "args":{**datasources_info["Birdclef"]}}
    TRAFFIC = {"class": ProvinceDataSource, "args":{**datasources_info["Province"], "name":"交通噪声"}}
    NATURE = {"class": ProvinceDataSource, "args":{**datasources_info["Province"], "name":"自然噪声"}}
    INDUSTRIAL = {"class": ProvinceDataSource, "args":{**datasources_info["Province"], "name":"工业噪声"}}
    SOCIAL = {"class": ProvinceDataSource, "args":{**datasources_info["Province"], "name":"社会噪声"}}
    CONSTRUCTIONAL = {"class": ProvinceDataSource, "args":{**datasources_info["Province"], "name":"建筑施工噪声"}}
    
@lru_cache(maxsize=len(SupportedSourceTypes))
def get_data_source(source_type: SupportedSourceTypes)->DataSourceBase:
    '''
    按需获取指定数据源的实例
    
    :param source_type: 数据源类型
    :return: 数据源实例
    '''
    return source_type.value["class"](**source_type.value["args"])
