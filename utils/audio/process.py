import torch
import torchaudio.transforms as T
import torchaudio.functional as F

def resample(signal:torch.Tensor, origin_sr, target_sr):
    """重采样：使用torchaudio里面的functional"""
    return F.resample(signal, origin_sr, target_sr)

def get_resample_transformer(origin_sr, target_sr):
    """重采样Transformer：返回一个用于resample的transformer"""
    return T.resample(origin_sr, target_sr, dtype=torch.float32)

def mix_down(signal:torch.Tensor):
    """音频信号声道融合"""
    if signal.shape[0]>1:
        return torch.mean(signal, dim=0, keepdim=True)
    return signal

def cut_signal(signal:torch.Tensor, num_samples, start=0):
    """裁剪音频信号到指定长度"""
    if signal.shape[1]>num_samples:
        signal = signal[:, start:,start+num_samples]
    return signal

def right_pad_signal(signal:torch.Tensor, num_samples):
    """填充音频信号到指定长度"""
    if (s_len:=signal.shape[1])<num_samples:
        paddings = num_samples - s_len
        return torch.nn.functional.pad(signal, (0, paddings)) # 用0填充paddings个位置
    return signal

def clip_signal(signal:torch.Tensor, start:int, end:int):
    '''按照start和end裁剪音频信号'''
    samples = signal.shape[1]
    if start<0:
        start = 0
    if end>samples:
        end = samples
    return signal[:, start:end]