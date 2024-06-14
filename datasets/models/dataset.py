import torch.nn as nn
from .base import DatasetBase

class Dataset(DatasetBase):
    def __init__(self, name:str, target_sr:int, duration:float, input_files:list[str], output_targets:list[int], extractor:nn.Module=None):
        '''
        __init__ function of Dataset class.
        Args:
            name (str): Name of the dataset.
            target_sr (int): Target sampling rate of the audio files.
            duration (float): Duration of the audio files in seconds.
            input_files (list[str]): List of input audio files.
            output_targets (list[int]): List of output targets.
            extractor (nn.Module, optional): Extractor module for feature extraction. Defaults to None.
        '''

        self.name = name
        assert len(input_files) == len(output_targets), "Number of input files and output targets must be equal"
        self.length = len(input_files)
        self.input_files = input_files
        self.output_targets = output_targets
        super().__init__(target_sr, duration, extractor=extractor)

    def __len__(self):
        return self.length
    
    def _get_audio_path(self, index):
        return self.input_files[index]
    
    def _get_label(self, index):
        return self.output_targets[index]
    