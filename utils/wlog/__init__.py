# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-06-11 10:55:28
# Description: This file contains the implementation of the WandbLogger class.

from ..plot import plt2ndarray

import wandb
from matplotlib.figure import Figure
from numpy import ndarray
import pandas as pd

def plt2image(fig: Figure) -> wandb.Image:
    
    img:ndarray = plt2ndarray(fig)
    
    if img.size[-1] == 4:
        mode = "RGBA"
    elif img.size[-1] == 3:
        mode = "RGB"
    else:
        raise ValueError("Invalid image format")
    
    return wandb.Image(img, mode=mode)

def df2table(df: pd.DataFrame) -> wandb.Table:
    # 创建一个 wandb.Table 对象，并指定表头
    columns = list(df.columns)
    tb = wandb.Table(columns=columns)
    
    # 将 DataFrame 的每一行数据添加到 wandb.Table 中
    for i, row in df.iterrows():
        tb.add_data(*row.tolist())
    
    return tb