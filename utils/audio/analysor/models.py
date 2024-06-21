# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-06-20 14:02:14
# Description: audio analysis module

import torch
import torchaudio
import numpy as np
from pydantic import BaseModel, Field
from matplotlib.figure import Figure

from ..features import FeatureType, get_feature_transformer
from utils import config

FEATURE_PARAMS = config.features

class AudioData(BaseModel):
    signal: torch.Tensor = Field(description="audio signal tensor")
    sample_rate: int = Field(description="sampling rate of the signal")
    duration: float = Field(description="duration of the signal in seconds")
    num_channels: int = Field(description="number of channels in the signal")
    
class AnalsisResult(AudioData):
    waveform_fig: Figure = Field(default=None, description="waveform figure of the signal")
    features_fig: Figure = Field(default=None, description="features of the signal")
    LeqA: float = Field(default=None, description="loudness of the signal in A-weighted scale")


