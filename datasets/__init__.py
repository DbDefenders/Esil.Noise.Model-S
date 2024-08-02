from .models import Dataset, Label, Category
from torch import nn
import torch
from utils.audio.extractor import EventExtractor
from torchvision import transforms
import audiomentations
import albumentations
class DatasetFactory:
    def __init__(self, category:Category, test_ratio:float=0.2, seed:int=1202):
        '''
        初始化数据集工厂

        :param category: 数据集类别
        :param test_ratio: 测试集比例
        :param seed: 随机种子
        '''
        self.category = category
        self.name = category.name
        self.test_ratio = test_ratio
        self.seed = seed

        self.X_train, self.X_test, self.y_train, self.y_test = category.get_train_test_data(test_ratio=test_ratio, seed=seed)

    def count_train_test_data(self) -> tuple:
        '''
        计数训练集和测试集样本数
        '''
        counter_train, counter_test = self.category.count_train_test_data(y_train=self.y_train, y_test=self.y_test)

        return counter_train, counter_test

    def create_dataset(self, *, train:bool, target_sr:int, duration:float, extractor:nn.Module, event_extractor:EventExtractor=None,
                       wav_augmentor:audiomentations.Compose=None, spec_augmentor:albumentations.Compose=None,
                       transform:transforms.Compose=None,device='cpu') -> torch.utils.data.Dataset:
        '''
        获取训练集或测试集数据集

        :param train: 是否为训练集
        :param target_sr: 目标采样率
        :param duration: 目标时长
        :param extractor: 特征提取器
        :return: 数据集
        '''
        if train:
            X = self.X_train
            y = self.y_train
        else:
            X = self.X_test
            y = self.y_test
        
        return Dataset(name=self.name, target_sr=target_sr, duration=duration, extractor=extractor, device=device, input_files=X, output_targets=y, 
                       event_extractor=event_extractor,wav_augmentor=wav_augmentor, spec_augmentor=spec_augmentor, transform=transform)
    
__all__ = ['DatasetFactory', 'Dataset', 'Label', 'Category']