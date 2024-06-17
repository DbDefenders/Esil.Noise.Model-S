from torch import nn
from abc import ABC, abstractmethod
from .panns import CNN10, CNN14

class Middleware(nn.Module):
    def __init__(self):
        super(Middleware, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x) # torch.Size([2, 1, 512, 431])
        x = self.pool(x) # torch.Size([2, 1, 256, 215])
        x = self.relu(x) # torch.Size([2, 1, 256, 215])
        return x
    
class CustomModelBase(ABC,nn.Module):
    def __init__(self, num_classes:int, input_height:int):
        super(CustomModelBase, self).__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.middleware = Middleware()
       
    @abstractmethod 
    def forward(self, x):
        pass

class MyCNN10(CustomModelBase):
    def __init__(self, num_classes:int, input_height:int):
        super(MyCNN10, self).__init__(num_classes, input_height)
        self.cnn = CNN10(num_classes, input_height)
        
    def forward(self, x):
        x = self.middleware(x)
        x = self.cnn(x)
        return x
    
    def __repr__(self):
        return 'my cnn 10'
    
class MyCNN14(CustomModelBase):
    def __init__(self, num_classes:int, input_height:int):
        super(MyCNN14, self).__init__(num_classes, input_height)
        self.cnn = CNN14(num_classes, input_height)
        
    def forward(self, x):
        x = self.middleware(x)
        x = self.cnn(x)
        return x
    
    def __repr__(self):
        return 'my cnn 14'