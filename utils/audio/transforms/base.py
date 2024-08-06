from abc import ABC, abstractmethod

class TransfromBase(ABC):
    @abstractmethod
    def process(self, data):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass