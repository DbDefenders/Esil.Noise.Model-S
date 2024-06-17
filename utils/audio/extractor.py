import torch
from typing import List
from functools import cached_property

# 导入特征提取相关的函数和类
from .features import FeatureType, get_spectrogram_transformer, get_melspec_transformer, get_mfcc_transformer, get_lfcc_transformer
from ..math.normalization import min_max_normalize
from ..math.relu import percent_relu

# 定义一个音频特征提取器类，继承自torch.nn.Module
class Extractor(torch.nn.Module):
    def __init__(
        self, 
        sample_rate:int,  # 音频采样率
        features:List[FeatureType],  # 特征类型列表
        normalize_func:callable=min_max_normalize,  # 归一化函数
        limit:int=10,  # 用于ReLU激活函数的阈值
        rep:int=0  # 未知用途，可能是用于调试或特定的重复次数
    ):
        # 确保至少选择了一种特征类型
        assert len(features)>0, "select_features should not be empty"
        super(Extractor, self).__init__()  # 调用父类的初始化方法
        self.hop_length = 512  # STFT窗口的跳跃长度，影响时间轴上的分辨率
        self.spec_n_fft = 1024  # STFT的FFT点数
        self.xfcc_n_fft = 2048  # eXtended Feature Cepstral Coefficients (XFCC)的FFT点数
        self.n_xfcc = 512  # XFCC的数量
        self.n_mels = 512  # Mel频率倒谱系数的数量
        self.win_length = None  # STFT窗口的长度，如果为None，则默认与n_fft相同
        self.sample_rate = sample_rate  # 音频采样率
        self.select_features = features  # 选择的特征类型列表
        self.normalize_func = normalize_func  # 归一化函数
        self.limit = limit  # 用于ReLU激活函数的阈值
        self.rep = rep  # 未知用途，可能是用于调试或特定的重复次数
        
    # 定义一个属性，用于获取spectrogram变换器
    @property
    def spectrogram(self)->callable:
        return get_spectrogram_transformer(self.spec_n_fft, self.hop_length, self.win_length)
    
    # 定义一个属性，用于获取mel spectrogram变换器
    @property
    def mel_spec(self)->callable:
        return get_melspec_transformer(self.sample_rate, self.xfcc_n_fft, self.n_mels, self.hop_length)
    
    # 定义一个属性，用于获取MFCC变换器
    @property
    def mfcc(self)->callable:
        return get_mfcc_transformer(self.sample_rate, self.n_xfcc, self.xfcc_n_fft, self.n_mels, self.hop_length)
    
    # 定义一个属性，用于获取LFCC变换器
    @property
    def lfcc(self)->callable:
        return get_lfcc_transformer(self.sample_rate, self.n_xfcc, self.xfcc_n_fft, self.hop_length, self.win_length)
    
    # 使用cached_property装饰器定义一个属性，用于存储特征映射
    @cached_property
    def features_map(self):
        return {
            FeatureType.SPECTROGRAM: self.spectrogram,
            FeatureType.MEL_SPECTROGRAM: self.mel_spec,
            FeatureType.MFCC: self.mfcc,
            FeatureType.LFCC: self.lfcc
        }

    # 前向传播方法，用于提取音频特征
    def forward(self, x):
        select_features = []  # 初始化一个列表，用于存储提取的特征
        for f in self.select_features:  # 遍历选择的特征类型
            select_features.append(self.features_map[f](x))  # 使用对应的变换器提取特征，并添加到列表中
                
        x_len = min([f.shape[1] for f in select_features])  # 获取所有特征中最小的时间轴长度

        for idx in range(len(select_features)):
            select_features[idx] = select_features[idx][:, :x_len, :]  # 截断特征，使其具有相同的时间轴长度
            select_features[idx] = percent_relu(select_features[idx],self.limit)  # 使用ReLU激活函数和阈值进行归一化

        x = torch.stack(select_features, dim=0)  # 将特征堆叠成一个张量，形状为[特征数量, 通道数, 频率轴长度, 时间轴长度]
        x = torch.squeeze(x)  # 移除维度为1的轴，形状变为[特征数量, 频率轴长度, 时间轴长度]
        return x  # 返回提取的特征张量
        
    # 定义一个字符串表示方法，用于打印提取器信息