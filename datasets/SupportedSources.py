
from .models.sources import US8KDataSource, ESC50DataSource, ProvinceDataSource,BirdclefDataSource

from utils import config

datasources_info = config.data_sources

US8K = US8KDataSource(**datasources_info['US8K'])
ESC50 = ESC50DataSource(**datasources_info['ESC50'])
BIRDCLEF = BirdclefDataSource(**datasources_info['Birdclef'])
TRAFFIC = ProvinceDataSource(name="交通噪声", **datasources_info['Province'])
NATURE = ProvinceDataSource(name="自然噪声", **datasources_info['Province'])
INDUSTRIAL = ProvinceDataSource(name="工业噪声", **datasources_info['Province'])
SOCIAL = ProvinceDataSource(name="社会噪声", **datasources_info['Province'])
CONSTRUCTIONAL = ProvinceDataSource(name="建筑施工噪声", **datasources_info['Province'])

__all__ = ['US8K', 'ESC50', 'TRAFFIC', 'NATURE', 'INDUSTRIAL', 'SOCIAL', 'CONSTRUCTIONAL','BIRDCLEF']