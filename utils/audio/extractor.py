import torch
import torchaudio
import librosa
from typing import List
from functools import cached_property

# 导入特征提取相关的函数和类
from .features import FeatureType, get_spectrogram_transformer, get_melspec_transformer, get_mfcc_transformer, get_lfcc_transformer
from ..math.normalization import min_max_normalize
from ..math.relu import percent_relu
from .analysor import find_sound_events, get_max_length_interval
from .process import clip_signal

class EventExtractor(torch.nn.Module):
    def __init__(self, sample_rate:int, duration:float, device, time_interval:float=0.5, n_fft:int=1024, hop_length:int=512, win_length:int=None):
        '''
        初始化音频事件提取器
        :param sample_rate: 音频采样率
        :param duration: 音频时长
        :param device: 计算设备
        :param time_interval: 事件时间间隔
        :param n_fft: STFT的FFT点数
        :param hop_length: STFT窗口的跳跃长度
        :param win_length: STFT窗口的长度，如果为None，则默认与n_fft相同
        '''
        self.device = device
        self.time_interval = time_interval
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft

    @cached_property
    def spectrogram_transform(self):
        return torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=2.0
        ).to(self.device)
    
    def forward(self, signal:torch.Tensor)->torch.Tensor:
        signal = signal.to(self.device)
        
        spectrogram = self.spectrogram_transform(signal)

        energy = torch.sum(spectrogram, dim=0)

        energy_threshold = torch.mean(energy)

        sound_frames = torch.where(energy > energy_threshold)[0]
        sound_times = librosa.frames_to_time(sound_frames.cpu().numpy(), sr=self.sample_rate, hop_length=self.hop_length)

        num_events, events = find_sound_events(sound_times, self.time_interval)
        
        if num_events == 0:
            # 没有找到任何事件, 返回原始信号
            return signal
        else:
            # 按事件裁剪
            start, end = get_max_length_interval(events)
            delta = end - start
            
            signal = clip_signal(signal, int(start*self.sample_rate), int(end*self.sample_rate))
            # 将信号长度加长
            multiplier = int(self.duration/delta)  # 计算增长比例
            signal = signal.repeat(1, multiplier)  # 重复信号
            return signal
            
            

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