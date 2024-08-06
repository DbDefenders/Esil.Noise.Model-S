# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-08-06 14:02:47
# Description: 针对频谱的数据增强方法

import torch
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np

def get_mask(
    start: int,
    end: int,
    total_length: int,
    batch_size: int = 1,
):
    """
    生成一个掩码，用于在指定的范围内随机选择序列的部分。

    参数：
    - start (int): 掩码长度的最小值（包含）。
    - end (int): 掩码长度的最大值（不包含）。
    - total_length (int): 输入序列的总长度。
    - batch_size (int): 批次大小，默认为1。

    返回：
    - mask (Tensor): 形状为 (batch_size, total_length) 的布尔张量，
      表示每个样本在总长度中被掩码的位置。
    """
    # 随机生成掩码长度，范围在 [start, end) 之间，形状为 (batch_size, 1)
    mask_length = torch.randint(start, end, (batch_size, 1))  # (batch_size, 1)

    # 随机生成掩码起始位置，确保不会超出总长度
    mask_position = torch.randint(
        0, max(1, total_length - mask_length.max()), (batch_size, 1)
    )  # (batch_size, 1)

    # 计算掩码结束位置
    mask_end = mask_position + mask_length  # (batch_size, 1)

    # 创建一个从 0 到 total_length-1 的张量，形状为 (1, total_length)
    arange = torch.arange(total_length).view(1, -1)  # (1, total_length)

    # 根据起始位置和结束位置生成掩码，掩码为布尔值
    mask = (mask_position <= arange) & (arange < mask_end)  # (batch_size, total_length)

    return mask  # 返回形状为 (batch_size, total_length) 的掩码


def get_time_freq_mask(
    channels: int,
    time_length: int,
    freq_length: int,
    time_mask_start: int = 0,
    time_mask_end: int = 10,
    freq_mask_start: int = 0,
    freq_mask_end: int = 8,
    num_mask: int = 1,
    mask_mode: Literal["intersection", "union"] = "intersection",
    device: torch.device = torch.device("cpu"),
):
    '''
    获取时频掩码，用于对频谱数据进行增强。
    - param: time_mask_start: 时间掩码的最小长度
    - param: time_mask_end: 时间掩码的最大长度
    - param: freq_mask_start: 频率掩码的最小长度
    - param: freq_mask_end: 频率掩码的最大长度
    - param: num_mask: 掩码的数量
    - param: mask_value: 掩码的值, 默认为0
    - param: mask_mode: 掩码模式，'intersection'表示交集，'union'表示并集
    '''
     # 初始化掩码
    mask = torch.zeros((channels, time_length, freq_length), dtype=torch.bool)

    for _ in range(num_mask):
        # 时间掩码
        time_mask = get_mask(
            start=time_mask_start,
            end=time_mask_end,
            total_length=time_length,
            batch_size=channels,
        ).unsqueeze(2)
        # 频率掩码
        freq_mask = get_mask(
            start=freq_mask_start,
            end=freq_mask_end,
            total_length=freq_length,
            batch_size=channels,
        )
        # 掩码
        if mask_mode == "intersection":
            current_mask = time_mask & freq_mask  # 取交集
        elif mask_mode == "union":
            current_mask = time_mask | freq_mask  # 取并集
        else:
            raise ValueError(f"Unsupported mask mode: {mask_mode}")

        # 累加多个掩码
        mask = mask | current_mask

    return mask.to(device)

def mask_data(data:torch.Tensor, mask:torch.Tensor, mask_value:int=0):
    '''
    对数据进行掩码，将掩码的位置替换为mask_value
    - param: data: 输入数据，形状为(channels, time_length, freq_length)
    - param: mask: 掩码，形状为(channels, time_length, freq_length)
    '''
    assert data.shape == mask.shape, f"data shape {data.shape} not match mask shape {mask.shape}"
    new_data = data.masked_fill_(mask, mask_value)
    return new_data
    
def mask2figure(mask, ax=None, grid:bool=False, width:int = 10):
    """
    绘制掩码

    - param: mask: 掩码张量，形状为 (time_length, freq_length)
    - return: Figure, 绘制的图像

    - Demo:
    >>> mask_tool = MaskDataProcess(num_mask=10, mask_mode='intersection')
    >>> mask = mask_tool.get_time_freq_mask(channels=1, time_length=100, freq_length=100)
    >>> fig = mask_tool.draw_mask(mask[0])
    >>> fig.show()
    """
    assert (
        len(mask.shape) == 2
    ), f"Unsupported mask shape: {mask.shape}, expected (time_length, freq_length)"

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    height = int(width / mask.shape[1] * mask.shape[0])

    ret = ax
    if ax is None:
        # 创建图像和坐标轴
        fig, ax = plt.subplots(figsize=(width, height))  # 设置图像大小
        ret = fig

    # 使用imshow绘制网格图
    cax = ax.imshow(mask, cmap="binary", interpolation="nearest")
    # fig.colorbar(cax, ax=ax)  # 添加颜色条

    if grid:
        # 设置网格线
        ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    # 隐藏x轴和y轴的刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ret