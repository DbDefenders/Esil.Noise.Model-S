import torch.nn as nn
import torchaudio
from .base import DatasetBase, resample, mix_down, cut_signal, right_pad_signal

from utils.audio.extractor import EventExtractor

class Dataset(DatasetBase):
    def __init__(self, name:str, target_sr:int, duration:float, input_files:list[str], output_targets:list[int], extractor:nn.Module=None, event_extractor:EventExtractor=None, device='cpu'):
        '''
        __init__ function of Dataset class.
        Args:
            name (str): Name of the dataset.
            target_sr (int): Target sampling rate of the audio files.
            duration (float): Duration of the audio files in seconds.
            input_files (list[str]): List of input audio files.
            output_targets (list[int]): List of output targets.
            extractor (nn.Module, optional): Extractor module for feature extraction. Defaults to None.
        '''

        self.name = name
        assert len(input_files) == len(output_targets), "Number of input files and output targets must be equal"
        self.length = len(input_files)
        self.input_files = input_files
        self.output_targets = output_targets
        super().__init__(target_sr, duration, extractor=extractor, device=device)
        self.event_extractor = event_extractor

    def __len__(self):
        return self.length
    
    def _get_audio_path(self, index):
        return self.input_files[index]
    
    def _get_label(self, index):
        return self.output_targets[index]
    
    def __getitem__(self, index):
        if self.event_extractor is not None:
            audio_file = self._get_audio_path(index)
            label = self._get_label(index)
            # 读取音频
            signal, sr = torchaudio.load(audio_file)
            signal =  signal.to(self.dtype)
            # 重采样
            signal = resample(signal, sr, self.sample_rate)
            # 声道融合
            signal = mix_down(signal)
            
            # ⭐事件提取
            signal = self.event_extractor.forward(signal)
            
            # 裁剪/填充音频
            signal = cut_signal(signal=signal, num_samples=self.num_samples) if signal.shape[1]>self.num_samples else right_pad_signal(signal=signal, num_samples=self.num_samples)
            # 提取音频特征
            if self.extractor is not None:
                feature = self.extractor(signal)
            else:
                feature = signal
            return feature, label, index
        else:
            return super().__getitem__(index)