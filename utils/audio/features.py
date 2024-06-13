from enum import Enum
import torchaudio.transforms as T
from typing import Union
from utils import config
class FeatureType(Enum):
    SPECTROGRAM = {'n_fft', 'hop_length', 'win_length'}
    MEL_SPECTROGRAM = {'sample_rate', 'n_fft', 'n_mels', 'hop_length', 'win_length'}
    MFCC = {'sample_rate', 'n_mfcc', 'n_fft', 'n_mels', 'hop_length'}
    LFCC = {'sample_rate', 'n_lfcc', 'n_fft', 'hop_length', 'win_length'}
    
    def __str__(self) -> str:
        return self.name.lower()
    
    
def get_feature_transformer(feature_type: Union[str, FeatureType], **kwargs):
    if isinstance(feature_type, str):
        feature_type = FeatureType[feature_type.upper()]
    if feature_type == FeatureType.SPECTROGRAM:
        return get_spectrogram_transformer(**kwargs)
    elif feature_type == FeatureType.MEL_SPECTROGRAM:
        return get_melspec_transformer(**kwargs)
    elif feature_type == FeatureType.MFCC:
        return get_mfcc_transformer(**kwargs)
    elif feature_type == FeatureType.LFCC:
        return get_lfcc_transformer(**kwargs)
    else:
        raise ValueError("Invalid feature type")

def get_spectrogram_transformer(n_fft, hop_length, win_length=None):
    return T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode='reflect',# 反射
        power=2.0
    )

def get_melspec_transformer(sample_rate, n_fft, n_mels, hop_length, win_length=None):
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        # onesided=True,#  Argument 'onesided' has been deprecated and has no influence on the behavior of this module.
        n_mels=n_mels,
        mel_scale="htk",
    )
    
def get_mfcc_transformer(sample_rate, n_mfcc, n_fft, n_mels, hop_length):
    return T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )

def get_lfcc_transformer(sample_rate, n_lfcc, n_fft, hop_length, win_length=None):
    return T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
    )

def feature_from_config(feature:FeatureType):

    return get_feature_transformer(feature,config.features[feature.name])