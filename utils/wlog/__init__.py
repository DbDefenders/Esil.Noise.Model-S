# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-06-11 10:55:28
# Description: This file contains the implementation of the WandbLogger class.


import wandb
from  enum import Enum

class LogFileType(Enum):
    Image = 1
    Audio = 2
    Table = 3
        
class WandbLogger():
    def __init__(self, *args):
        self.init_wandb(*args)
        pass
    
    def init_wandb(*args):
        pass

    def log_wandb(*args):
        pass
    def log_files(file_type:LogFileType, *args):
        pass

    def finish_wandb(*args):
        pass