from .models import DatasetBase, Label, Category

class Dataset(DatasetBase):
    def __init__(self, target_sr:int, duration:int, category:Category,test_ratio:float=0.2, seed:int=1202):
        super().__init__(target_sr=target_sr, duration=duration)
        self.category = category
        self.X_train, self.y_train, self.X_test, self.y_test = self.category.get_train_test_data(test_ratio=test_ratio, seed=seed)