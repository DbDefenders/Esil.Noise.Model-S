# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-08-06 14:06:36
# Description: 用于频谱的数据增强


from .base import TransfromBase
from typing import Literal
import torch
from functools import partial
from ..functional.feature import get_mask, mask2figure

class DataMasker(TransfromBase):
    '''时频掩蔽'''
    def __init__(
        self,
        time_mask_start: int = 0,
        time_mask_end: int = 10,
        freq_mask_start: int = 0,
        freq_mask_end: int = 8,
        num_mask: int = 1,
        mask_value: int = 0,
        mask_mode: Literal["intersection", "union"] = "intersection",
    ):
        """
        - param: time_mask_start: 时间掩码的最小长度
        - param: time_mask_end: 时间掩码的最大长度
        - param: freq_mask_start: 频率掩码的最小长度
        - param: freq_mask_end: 频率掩码的最大长度
        - param: num_mask: 掩码的数量
        - param: mask_value: 掩码的值, 默认为0
        - param: mask_mode: 掩码模式，'intersection'表示交集，'union'表示并集
        """
        self.time_mask_start = time_mask_start
        self.time_mask_end = time_mask_end
        self.freq_mask_start = freq_mask_start
        self.freq_mask_end = freq_mask_end
        self.num_mask = num_mask
        self.mask_value = mask_value
        self.mask_mode = mask_mode

        self.time_mask = partial(
            get_mask, start=self.time_mask_start, end=self.time_mask_end
        )
        self.freq_mask = partial(
            get_mask, start=self.freq_mask_start, end=self.freq_mask_end
        )

    def get_time_freq_mask(
        self,
        channels: int,
        time_length: int,
        freq_length: int,
        device: torch.device = "cpu",
    ):
        # 初始化掩码
        mask = torch.zeros((channels, time_length, freq_length), dtype=torch.bool)

        for _ in range(self.num_mask):
            # 时间掩码
            time_mask = self.time_mask(
                total_length=time_length, batch_size=channels
            ).unsqueeze(2)
            # 频率掩码
            freq_mask = self.freq_mask(
                total_length=freq_length, batch_size=channels
            ).unsqueeze(1)
            # 掩码
            if self.mask_mode == "intersection":
                current_mask = time_mask & freq_mask  # 取交集
            elif self.mask_mode == "union":
                current_mask = time_mask | freq_mask  # 取并集
            else:
                raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

            # 累加多个掩码
            mask = mask | current_mask

        return mask.to(device)
        
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """
        给data添加随机掩码
        """
        # 判断data的维度
        if len(data.shape) == 2:
            # 如果是2维，则添加一个通道维度
            data = data.unsqueeze(0)

        assert (
            len(data.shape) == 3
        ), f"Unsupported data shape: {data.shape}, expected (channels, time_length, freq_length) or (time_length, freq_length)"

        channels, time_length, freq_length = data.shape

        mask = self.get_time_freq_mask(channels, time_length, freq_length, device=data.device)

        data = data.masked_fill_(mask, self.mask_value)

        return data

    def __repr__(self):
        return f"DataMasker(time_mask_start={self.time_mask_start}, time_mask_end={self.time_mask_end}, freq_mask_start={self.freq_mask_start}, freq_mask_end={self.freq_mask_end}, num_mask={self.num_mask}, mask_value={self.mask_value}, mask_mode={self.mask_mode})"

    @staticmethod
    def draw_mask(mask, ax=None, grid:bool=False, width:int = 10):
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
        return mask2figure(mask=mask, ax=ax, grid=grid, width=width)