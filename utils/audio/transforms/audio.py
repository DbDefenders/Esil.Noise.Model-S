import torch
from typing import Literal

from .base import TransfromBase
from ..functional.audio import (
    resample,
    vad,
    change_speed_by_librosa,
    change_speed_by_numpy,
    adjust_volume,
    pad,
    clip,
    mix_down_channels,
    repeat_channels,
)


# region Resampler
class Resampler(TransfromBase):
    """音频重采样"""

    def __init__(
        self,
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
        self.sr = sr
        self.target_sr = target_sr
        self.res_type = res_type

    def process(self, data: torch.Tensor, sr: int = None) -> torch.Tensor:
        if sr == None:
            sr = self.sr
        return resample(data, sr, self.target_sr, self.res_type)

    def __repr__(self):
        return f"Resampler(sr={self.sr}, target_sr={self.target_sr}, res_type={self.res_type})"


# endregion


# region ChannelChanger
class ChannelChanger(TransfromBase):
    """通道融合或重复"""

    def __init__(self, num_channels: Literal[1, 2] = 1):
        self.num_channels = num_channels
    
    def process(self, data: torch.Tensor):
        channels, _ = data.shape
        if channels == self.num_channels:
            return data
        if channels > self.num_channels:
            return mix_down_channels(data)
        else:
            return repeat_channels(data, self.num_channels)

    def __repr__(self):
        return f"ChannelChanger(num_channels={self.num_channels})"


# endregion


# region Clipper
class Clipper(TransfromBase):
    """音频裁剪"""

    def __init__(self, duration: float, sr: int):
        self.duration = duration
        self.sr = sr
        self.samples = int(duration * sr)

    def process(self, data: torch.Tensor):
        new_data = clip(signal=data, length=self.samples)
        return new_data

    def __repr__(self):
        return f"Clipper(duration={self.duration}, sr={self.sr})"


# endregion


# region Padder
class Padder(TransfromBase):
    """音频填充"""

    def __init__(
        self,
        duration: float,
        sr: int,
        pad_value: float = 0.0,
        pad_side: Literal["left", "right", "both"] = "right",
    ):
        self.duration = duration
        self.sr = sr
        self.samples = int(duration * sr)
        self.pad_value = pad_value
        self.pad_side = pad_side

    def process(self, data: torch.Tensor):
        new_data = pad(
            signal=data,
            length=self.samples,
            pad_value=self.pad_value,
            pad_side=self.pad_side,
        )
        return new_data

    def __repr__(self):
        return f"Padder(duration={self.duration}, sr={self.sr}, pad_value={self.pad_value})"


# endregion


# region VadClipper
class VadClipper(TransfromBase):
    """静音裁剪"""

    def __init__(self, top_db: int = 20):
        self.top_db = top_db

    def process(self, data: torch.Tensor):
        new_data = vad(data, top_db=self.top_db)
        return new_data

    def __repr__(self):
        return f"VadClipper(top_db={self.top_db})"


# endregion


# region SpeedChanger
class SpeedChangerL(TransfromBase):
    """
    使用librosa.effects.time_stretch()函数实现变速变调，对比SpeedChangeN效果更好
    """

    def __init__(self, new_speed: float):
        """
        - param new_speed: 变速的速率，需要大于0，(0,1]为减速，(1,inf)为加速
        """
        assert new_speed > 0, "Speed rate should be greater than 0"

        self.new_speed = new_speed

    def process(self, data: torch.Tensor) -> torch.Tensor:
        new_data = change_speed_by_librosa(data, self.new_speed)
        return new_data

    def __repr__(self):
        return f"SpeedChangerL(new_speed={self.new_speed})"


class SpeedChangerN(TransfromBase):
    """
    使用numpy.interp()函数实现变速变调
    """

    def __init__(self, new_speed: float):
        """
        - param new_speed: 变速的速率，需要大于0，(0,1]为减速，(1,inf)为加速
        """
        assert new_speed > 0, "Speed rate should be greater than 0"

        self.new_speed = new_speed

    def process(self, data: torch.Tensor) -> torch.Tensor:
        new_data = change_speed_by_numpy(data, self.new_speed)
        return new_data

    def __repr__(self):
        return f"SpeedChangerN(new_speed={self.new_speed})"


# endregion


# region VolumeAdjuster
class VolumeAdjuster:
    """音量调节"""

    def __init__(self, target_db: float = -20):
        self.target_db = target_db

    def process(self, signal: torch.Tensor):
        return adjust_volume(signal, self.target_db)

    def __repr__(self):
        return f"VolumeAdjuster(target_db={self.target_db})"


# endregion
