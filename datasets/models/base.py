import os
from abc import ABC, abstractmethod
from typing import List, Union
from copy import deepcopy
import gc
from torchvision import transforms
import torch
import torchaudio
from utils.common import BisectList, get_child, save_json
from utils.audio.process import resample, mix_down, cut_signal, right_pad_signal
import audiomentations
import albumentations
import numpy as np

class DataSourceBase(BisectList, ABC):
    def __init__(self, base_dir:str, name:str, label:int=None, length:int=None, childs:List['DataSourceBase']=None):
        '''
        数据源基类
        Args:
            base_dir: 数据集根目录
            name: 数据集名称
            label: 数据集标签
            length: 数据集长度
            childs: 数据集子集
        '''
        self.base_dir = base_dir
        self.name = name
        self.label = label
        self._id = label
        if childs is None:
            childs = []
        
        super().__init__(lst_of_lst=childs, sort_key='label', reset_index='label', length=length)
        
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, _):
        raise AttributeError("不能修改标签的id！")
    
    @property
    def childs(self) -> List['DataSourceBase']:
        return self._lst_of_lst
    
    @property
    def childs_info(self) -> dict:
        return {c.name:len(c) for c in self.childs}
    
    def add_child(self, child: 'DataSourceBase'):
        self.set_lst_of_lst(self._lst_of_lst + [deepcopy(child)])
        
    def get_child(self, index_or_name:Union[str,int]) -> 'DataSourceBase':
        return deepcopy(get_child(index_or_name, self._lst_of_lst))
        
    def get_childs(self, index_or_lst:Union[str,int,List[Union[str,int]]]) -> List['DataSourceBase']:
        if isinstance(index_or_lst, list):
            return [self.get_child(index) for index in index_or_lst]
        else:
            return [self.get_child(index_or_lst)]

    def to_dict(self):
        ret = {
            'name': self.name,
            'label': self.label,
            'length': len(self)
        }
        if self.has_next():
            ret['childs'] =  [child.to_dict() for child in self.childs]
        return ret
    
    def check_validation(self):
        '''
        检查数据集是否存在重复文件，以及文件是否存在
        '''
        try:
            ret = False
            results = []
            not_exists = []
            for i in range(len(self)):
                results.append(self[i])
                if not os.path.exists(self[i]):
                    not_exists.append(self[i])
            if len(not_exists) > 0:
                raise FileNotFoundError(f"文件不存在：{','.join(not_exists)}")
            if not len(set(results)) == len(self):
                raise ValueError("数据集中存在重复文件！")
            ret = True
            return ret
        finally:
            del results
            del not_exists
            gc.collect()             
    
    @abstractmethod
    def get_file_path(self, index):
        raise NotImplementedError()
        
    def __getitem__(self, index):
        if index < 0: index += len(self)
        assert index < len(self), f"Index {index} out of range: max = {len(self)-1}"
        if self.has_next():
            return self.find_in_next_node(index)
        else:
            # return os.path.join(self.base_dir, self.get_file_path(index)), self.name, self.label
            return os.path.join(self.base_dir, self.get_file_path(index))
        
    def __eq__(self, other):
        return (self.name == other.name and self.base_dir == other.base_dir) or (self[0] == other[0] and self[-1] == other[-1])
    
    
    def __hash__(self):
        return hash(self.name+self.base_dir)
        
    def save_to_file(self, file_path):
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
        ret = self.to_dict()
        save_json(obj=ret, file_path=file_path)
        print('Saved DataSource to:', file_path)
        return file_path
    
    def to_label(self)->"Label":
        if self.has_next():
            return Label(name=self.name, sources=self.childs)
        else:
            return Label(name=self.name, sources=[self])
        
    
class Label(BisectList):
    def __init__(self, name, sources:Union[DataSourceBase, List[DataSourceBase]], id:int=None):
        self.name = name
        self.id = id
        if isinstance(sources, DataSourceBase):
            sources = [sources]
        super().__init__(lst_of_lst=sources, sort_key='name', reset_index=None)
    
    @property
    def sources(self):
        return self._lst_of_lst
    
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
        return self.find_in_next_node(index)
    
    def __repr__(self):
        return f"Label(id={self.id}, name='{self.name}', sources={self.sources})"
    
class DatasetBase(ABC, torch.utils.data.Dataset):
    '''
    audiofile -> signal,sr -> features -> extractor -> **dataset** -> dataloader -> model
    '''
    def __init__(self, target_sr:int, duration:float, *, extractor:torch.nn.Module=None,dtype=torch.float32,
                 transform:transforms.Compose=None,wav_augmentor:audiomentations.Compose=None, spec_augmentor:albumentations.Compose=None, device='cpu'):
        self.target_sr = target_sr
        self.duration = duration
        self.device = device
        self.dtype = dtype
        self.extractor = extractor
        self.wav_augmentor = wav_augmentor
        self.spec_augmentor = spec_augmentor
        self.transform = transform
    
    @abstractmethod
    def _get_audio_path(self, index):
        pass
    
    @abstractmethod
    def _get_label(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @property
    def num_samples(self):
        return int(self.target_sr * self.duration)
    
    @property
    def sample_rate(self):
        return self.target_sr
    
    def __getitem__(self, index):
        audio_file = self._get_audio_path(index)
        label = self._get_label(index)
        # 读取音频
        signal, sr = torchaudio.load(audio_file)
        signal = signal.to(self.dtype)
        
        # 重采样
        signal = resample(signal, sr, self.sample_rate)
        # 声道融合
        signal = mix_down(signal)
        # 裁剪/填充音频
        signal = cut_signal(signal=signal, num_samples=self.num_samples) if signal.shape[1]>self.num_samples else right_pad_signal(signal=signal, num_samples=self.num_samples)
        # 波形增强
        if self.wav_augmentor is not None:
            signal = torch.tensor(self.wav_augmentor(signal.numpy(),sr))
        signal = signal.to(self.device)

        feature = signal
        # 提取音频特征
        if self.extractor is not None:
            feature = self.extractor(signal)
        
        # 频谱增强
        if self.spec_augmentor is not None:
            # 将特征转换为numpy数组，并调整轴的顺序以匹配spec_augmentor的输入要求
            feature = np.array(feature.cpu()).transpose(1, 2, 0)
            # 应用频谱增强
            feature = self.spec_augmentor(image=feature)["image"]
            # 将特征转回原始形状，并转换为tensor
            feature = torch.tensor(feature.transpose(2, 0, 1))

        # 特征转换
        if self.transform is not None:
            feature = self.transform(feature)
            
        return feature, label, index

    
    
    