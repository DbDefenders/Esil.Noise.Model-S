from typing import Union
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import torchaudio.transforms as T
from PIL import Image
from io import BytesIO
import wandb
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 统一修改字体
plt.rcParams['font.family'] = ['STSong']

FIG_SIZE = (8,4)

def get_spectrogram(n_fft=512):
    return T.Spectrogram(n_fft=n_fft)


def get_mel_spectrogram(sample_rate:int, n_fft=1024, win_length=None, hop_length=512, n_mels=128):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    return mel_spectrogram

def a_weighting(frequencies):
    '''
    A-weighting function to compensate for the perceptual effect of human hearing.
    '''
    ra = (12200**2 * frequencies**4) / (
        (frequencies**2 + 20.6**2) *
        (frequencies**2 + 12200**2) *
        np.sqrt((frequencies**2 + 107.7**2) * (frequencies**2 + 737.9**2))
    )
    a_weight = 2.00 + 20 * np.log10(ra)
    return a_weight

def plot_frequency_spectrogram(signal:Union[np.ndarray, torch.Tensor], sr:int):
    if isinstance(signal, torch.Tensor):
        signal = signal.mean(axis=0).flatten().numpy()
        
    # 2. 进行傅里叶变换
    D = np.abs(librosa.stft(signal))

    # 3. 计算频谱
    # 取平均值，得到每个频率的平均幅度
    magnitude_spectrum = np.mean(D, axis=1)

    # 4. 频率轴
    frequencies = np.linspace(0, sr / 2, len(magnitude_spectrum))

    # 5. 应用A加权曲线
    a_weight = a_weighting(frequencies)
    a_weighted_spectrum = magnitude_spectrum * 10**(a_weight / 20)

    # 6. 转换为分贝值
    # 分贝值 = 20 * log10(幅度)
    db_spectrum = 20 * np.log10(a_weighted_spectrum + 1e-6)  # 加上一个小值以避免 log(0)

    # 7. 计算Leq
    Leq = 10 * np.log10(np.mean(a_weighted_spectrum**2) + 1e-6)
    
    # 找到最大分贝值及其对应的频率
    max_db_index = np.argmax(db_spectrum)
    max_db_value = db_spectrum[max_db_index]
    max_db_frequency = frequencies[max_db_index]
    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(frequencies, db_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Spectrum')
    plt.axvline(x=max_db_frequency, color='r', linestyle='--')
    plt.text(max_db_frequency+1000, max_db_value, f'{max_db_frequency:.0f} Hz:{max_db_value:.2f} dB', color='red', ha='right')
    return fig, Leq
    
    
def plot_waveform(waveform:Union[torch.Tensor, np.ndarray], sr, title="Waveform", ax=None):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        fig, ax = plt.subplots(num_channels, 1, figsize=FIG_SIZE)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    return fig

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    return fig


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1, figsize=FIG_SIZE)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    return fig

def plt2ndarray(figure):
    buf = BytesIO() 
    figure.savefig(buf, format="jpg") # 将 Matplotlib 图像保存到内存中的一个缓冲区
    buf.seek(0)
    image = Image.open(buf) # 使用 Pillow 打开缓冲区中的图像
    image_array = np.array(image) # 将 Pillow 图像转换为 NumPy 数组
    buf.close() # 关闭缓冲区
    return image_array # jpg: RGB, png: RGBA

def plt2wandbImage(figure):
    arr = plt2ndarray(figure)
    return wandb.Image(arr, mode="RGB")