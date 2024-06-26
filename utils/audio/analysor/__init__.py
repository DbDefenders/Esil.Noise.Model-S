import torch
import torchaudio
import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from utils.audio.features import FeatureType, get_feature_transformer
from utils.audio.process import mix_down
from utils.plot import plot_spectrogram, plot_events, plot_waveform
from utils import config
# from .models import AnalsisResult

WIDTH = 8
HEIGHT = 3

def find_sound_events(arr, interval):
    if list(arr)==0:
        return 0, []

    intervals = []
    current_interval = [arr[0]]

    for i in range(1, len(arr)):
        if abs(arr[i] - arr[i - 1]) < interval:
            current_interval.append(arr[i])
        else:
            intervals.append(current_interval)
            current_interval = [arr[i]]

    intervals.append(current_interval)

    result = []
    for i in range(len(intervals)):
        result.append([intervals[i][0], intervals[i][-1]])

    return len(intervals), result

def get_max_length_interval(intervals):
    max_length = 0
    start = 0 
    end = 0
    for interval in intervals:
        if interval[-1] - interval[0] > max_length:
            max_length = interval[-1] - interval[0]
            start = interval[0]
            end = interval[-1]
    return start, end

def get_features_fig(signal: torch.Tensor, feature_params:dict=None) -> Figure:
    '''
    Get a figure of all features of the given signal.
    '''
    if signal.shape[0]>1:
        signal = mix_down(signal)

    if feature_params is None:
        feature_params = config.features

    fig_num = len(feature_params)
    fig, axes = plt.subplots(nrows=fig_num, ncols=1, figsize=(WIDTH, HEIGHT * fig_num))

    for idx, feature_name in enumerate(feature_params.keys()):
        feature_param = feature_params[feature_name]
        feature_type = FeatureType[feature_name]
        transfomer = get_feature_transformer(feature_type, **feature_param)
        feature = transfomer(signal)
        axes[idx] = plot_spectrogram(feature[0], title=feature_name, ax=axes[idx])
    fig.tight_layout()
    return fig

def get_waveform_fig(signal: torch.Tensor, sample_rate: int, interval:float=0.5)-> Figure:
    '''
    Get a figure of the waveform of the given signal.
    '''
    if signal.shape[0]>1:
        signal = mix_down(signal)

    duration = signal.shape[1] / sample_rate

    fig_num = 4
    fig, axes = plt.subplots(nrows=fig_num, ncols=1, figsize=(WIDTH, HEIGHT * fig_num))

    # waveform
    axes[0] = plot_waveform(signal, sample_rate, ax=axes[0])
    axes[0].set_xlim(0, duration)

    signal = signal.flatten().numpy()

    # 计算短时能量
    frame_length = 1024
    hop_length = 512
    energy = np.array([
        sum(abs(signal[i:i+frame_length]**2))
        for i in range(0, len(signal), hop_length)
    ])

    # 转换为分贝
    ref_energy = np.max(energy)
    ref_energy = 2e-5 * ref_energy
    energy_db = 10 * np.log10(energy / ref_energy)

    # 时间轴
    time_arr = librosa.frames_to_time(np.arange(len(energy)), sr=sample_rate, hop_length=hop_length)

    # 画能量谱
    axes[1].plot(time_arr, energy_db, label='Calculated dB from signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Energy (dB)')
    axes[1].set_title('Short-time Energy vs Time')
    axes[1].set_xlim(0, duration)
    axes[1].legend()

    # 能量阈值
    energy_threshold = np.mean(energy)

    # 画声音集中的部份
    axes[2].plot(librosa.times_like(energy, sr=sample_rate, hop_length=hop_length), energy, label='Energy')
    axes[2].axhline(y=energy_threshold, color='r', linestyle='--', label='Energy threshold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('Energy vs Time')
    axes[2].set_xlim(0, duration)
    axes[2].legend()

    # 找到声音集中的部份
    sound_frames = np.where(energy > energy_threshold)[0]
    sound_times = librosa.frames_to_time(sound_frames, sr=sample_rate, hop_length=hop_length)

    # 事件时间间隔(s)
    time_interval = interval
    
    num_events, events = find_sound_events(sound_times, time_interval)
    # start, end = get_max_length_interval(events)
    print(f'Number of sound events: {num_events}')
    axes[3] = plot_events(events, ax=axes[3])
    axes[3].set_xlabel('Time (s)')
    axes[3].set_xlim(0, duration)

    fig.tight_layout()
    return fig

