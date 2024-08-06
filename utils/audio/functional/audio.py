# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-08-06 14:02:09
# Description: 针对音频信号的处理方法

import numpy as np
import torch
import librosa
import torchaudio.functional as taf
from typing import Literal

# region resample
RESAMPLE_TYPE = {
    'librosa':['soxr_vhq','soxr_mq','soxr_lq','soxr_qq', 'kaiser_best', 'kaiser_fast', 'fft','scipy', 'polyphase', 'linear', 'zero_order_hold','sinc_best','sinc_medium','sinc_fastest'],
    'torchaudio':['sinc_interp_hann','sinc_interp_kaiser']
}

def resample(
    signal: torch.Tensor,
    sr: int,
    target_sr: int,
    res_type: Literal[
        "soxr_vhq",
        "soxr_hq",
        "soxr_mq",
        "soxr_lq",
        "soxr_qq",
        "kaiser_best",
        "kaiser_fast",
        "fft",
        "scipy",
        "polyphase",
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_medium",
        "sinc_fastest",
        "sinc_interp_hann",
        "sinc_interp_kaiser",
    ] = "kaiser_best",
):
    if res_type in RESAMPLE_TYPE['librosa']:
        return resample_by_librosa(signal, sr, target_sr, res_type)
    elif res_type in RESAMPLE_TYPE['torchaudio']:
        return resample_by_torchaudio(signal, sr, target_sr, res_type)
    else:
        raise ValueError("Unsupported resampling method")

def resample_by_librosa(
    signal: torch.Tensor,
    sr: int,
    target_sr: int,
    res_type: Literal[
        "soxr_vhq",
        "soxr_hq",
        "soxr_mq",
        "soxr_lq",
        "soxr_qq",
        "kaiser_best",
        "kaiser_fast",
        "fft",
        "scipy",
        "polyphase",
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_medium",
        "sinc_fastest",
    ] = "kaiser_best",
) -> torch.Tensor:
    """
    使用librosa库对信号进行重采样
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param sr: int > 0, 输入信号的采样率
    - param target_sr: int > 0, 输出信号的采样率
    - param res_type: 重采样使用的滤波器类型，默认为'soxr_hq'; 可选值:
        'soxr_vhq', 'soxr_hq', 'soxr_mq' or 'soxr_lq'
            `soxr` Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation.
            ``'soxr_hq'`` is the default setting of `soxr`.
        'soxr_qq'
            `soxr` Quick cubic interpolation (very fast, but not bandlimited)
        'kaiser_best'
            `resampy` high-quality mode
        'kaiser_fast'
            `resampy` faster method
        'fft' or 'scipy'
            `scipy.signal.resample` Fourier method.
        'polyphase'
            `scipy.signal.resample_poly` polyphase filtering. (fast)
        'linear'
            `samplerate` linear interpolation. (very fast, but not bandlimited)
        'zero_order_hold'
            `samplerate` repeat the last value between samples. (very fast, but not bandlimited)
        'sinc_best', 'sinc_medium' or 'sinc_fastest'
            `samplerate` high-, medium-, and low-quality bandlimited sinc interpolation.

    - return: torch.Tensor, 输出信号，形状为(channels, length*target_sr/sr)
    """
    assert sr > 0 and target_sr > 0, "Sampling rate must be positive"
    channels, _ = signal.shape

    samples = []

    for i in range(channels):
        samples.append(
            librosa.resample(
                signal[i].numpy(), orig_sr=sr, target_sr=target_sr, res_type=res_type
            )
        )

    new_signal = torch.tensor(samples)

    return new_signal

def resample_by_torchaudio(
    signal: torch.Tensor,
    sr: int,
    target_sr: int,
    res_type: Literal["sinc_interp_hann", "sinc_interp_kaiser"] = "sinc_interp_hann",
) -> torch.Tensor:
    new_signal = taf.resample(
        signal, orig_freq=sr, new_freq=target_sr, resampling_method=res_type
    )
    return new_signal
# endregion

# region speed change
def change_speed_by_librosa(
    signal: torch.Tensor,
    speed_rate: float,
) -> torch.Tensor:
    """
    使用librosa.effects.time_stretch()函数实现变速变调
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param speed_rate: float > 0, 变速的速率，需要大于0，(0,1]为减速，(1,inf)为加速
    - return: torch.Tensor, 输出信号，形状为(channels, length*speed_rate)
    """
    assert speed_rate > 0, "Speed rate should be greater than 0"

    new_signal = librosa.effects.time_stretch(signal.numpy(), rate=speed_rate)
    new_signal = torch.from_numpy(new_signal).float()

    return new_signal

def change_speed_by_numpy(
    signal: torch.Tensor,
    speed_rate: float,
) -> torch.Tensor:
    """
    使用numpy.interp()函数实现变速变调
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param speed_rate: float > 0, 变速的速率，需要大于0，(0,1]为减速，(1,inf)为加速
    - return: torch.Tensor, 输出信号，形状为(channels, length*speed_rate)
    """
    assert speed_rate > 0, "Speed rate should be greater than 0"

    channels, length = signal.shape
    new_length = int(length * speed_rate)
    new_signal = torch.zeros((channels, new_length))
    for i in range(channels):
        
        new_signal[i] = torch.from_numpy(
            np.interp(
                np.linspace(start=0, stop=length, num=new_length),
                np.arange(0, length), 
                signal[i].numpy()
            )
        ).float()

    return new_signal
# endregion

# region vad
def vad(signal:torch.Tensor, top_db:int=20):
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(0) # 增加通道维 (length,)->(1,length)
        
    intervals = librosa.effects.split(signal, top_db=top_db) # 所有大于top_db的区间
    
    if len(intervals) == 0:
        return signal # 全为静音，直接返回原信号
    
    samples = []
    for start, end in intervals:
        samples.append(signal[:, start:end])
        
    new_signal = torch.cat(samples, dim=1) # 合并区间
    
    return new_signal
# endregion

# region volume adjustment
def get_rms_db(signal: torch.Tensor):
    """
    计算信号的均方根值，并转换为分贝单位
    """
    mean_square = torch.mean(signal**2)
    if mean_square.item() == 0:
        return 0
    return 10 * torch.log10(mean_square).item()

def adjust_volume(signal: torch.Tensor, target_db: float):
    """
    调整音量到目标分贝值
    """
    current_db = get_rms_db(signal)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    signal = signal * gain
    return signal
# endregion

# region channel mix down / repeat
def mix_down_channels(signal: torch.Tensor):
    """
    将多通道信号合并为单通道信号
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    """
    channels, _ = signal.shape
    if channels == 1:
        return signal
    else:
        return torch.mean(signal, dim=0, keepdim=True)

def repeat_channels(signal: torch.Tensor, repeat_num: int):
    """
    将单通道信号重复多次
    - param signal: torch.Tensor, 输入信号，形状为(1, length)
    - param repeat_num: int > 0, 重复的次数
    """
    assert repeat_num > 0, "Repeat number should be greater than 0"
    channels, _ = signal.shape
    assert channels == 1, "Only support single channel signal"
    if repeat_num == 1:
        return signal
    else:
        return signal.repeat(repeat_num, 1)
# endregion

# region clip / pad
def clip(signal: torch.Tensor, length: int):
    """
    裁剪信号
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param length: int > 0, 裁剪的长度
    """
    _, signal_length = signal.shape
    if signal_length <= length:
        return signal
    else:
        return signal[:, :length]

def pad(signal: torch.Tensor, length: int, pad_value: float=0, pad_side: Literal['left', 'right', 'both']='right'):
    '''
    填充信号
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param length: int > 0, 填充后的长度
    - param pad_value: float, 填充值
    - param pad_side: str, 填充位置，可选值：'left', 'right', 'both', 默认为'right'
    '''
    _, signal_length = signal.shape
    if signal_length >= length:
        return signal
    new_signal = torch.zeros((signal.shape[0], length)) + pad_value
    if pad_side == 'right':
        new_signal[:, :signal_length] = signal
    elif pad_side == 'left':
        new_signal[:, -signal_length:] = signal
    elif pad_side == 'both':
        new_signal[:, (length - signal_length) // 2: (length - signal_length) // 2 + signal_length] = signal
    else:
        raise ValueError("Unsupported pad side")
    return new_signal
# endregion

# region noise reduction/addition (待测试)
def add_noise(signal: torch.Tensor, noise_signal: torch.Tensor, noise_db: float):
    """
    增加噪声
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param noise_signal: torch.Tensor, 噪声信号，形状为(channels, length)
    - param noise_db: float > 0, 噪声信号的分贝值
    """
    signal_db = get_rms_db(signal)
    noise_db = get_rms_db(noise_signal)
    if signal_db > noise_db:
        return signal
    else:
        return signal + (noise_db - signal_db) / 20 * noise_signal
    
def reduce_noise(signal: torch.Tensor, noise_signal: torch.Tensor, noise_db: float):
    """
    降低噪声
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param noise_signal: torch.Tensor, 噪声信号，形状为(channels, length)
    - param noise_db: float > 0, 噪声信号的分贝值
    """
    signal_db = get_rms_db(signal)
    noise_db = get_rms_db(noise_signal)
    if signal_db < noise_db:
        return signal
    else:
        return signal - (noise_db - signal_db) / 20 * noise_signal
# endregion

# region pitch shift (待测试)
def pitch_shift(signal: torch.Tensor, n_steps: int):
    """
    音调变换
    - param signal: torch.Tensor, 输入信号，形状为(channels, length)
    - param n_steps: int, 音调变换的半音数，正数为升高音调，负数为降低音调
    """
    return librosa.effects.pitch_shift(signal.numpy(), sr=16000, n_steps=n_steps)
# endregion