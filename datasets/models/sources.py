import os
import json
import pandas as pd
from utils.common import create_repr_str
from .base import DataSourceBase

class ESC50DataSource(DataSourceBase):
    def __init__(
        self,
        base_dir: str,
        meta_file: str = "static/meta_file/esc50.csv",
        dataframe: pd.DataFrame = None,
        name: str = "ESC50",
        label: int = None,
        length: int = None,
    ):
        self.meta_file = meta_file
        childs = None
        if dataframe is None:
            dataframe = pd.read_csv(meta_file)
        self.__dataframe__ = dataframe
        annotations = self.__dataframe__
        annotations["target"] = annotations["target"].astype(int)
        annotations = (
            annotations[annotations["target"] == label]
            if label is not None
            else annotations
        )
        classes = annotations["target"].unique()
        if len(classes) > 1:
            classes.sort()
            childs = []
            for t in classes:
                tmp = annotations[annotations["target"] == t]
                childs.append(
                    ESC50DataSource(
                        meta_file=meta_file,
                        base_dir=base_dir,
                        dataframe=self.__dataframe__,
                        name=tmp.iloc[0]["category"],
                        label=t,
                        length=len(tmp),
                    )
                )

        super().__init__(
            base_dir=base_dir, name=name, label=label, length=length, childs=childs
        )

    def get_file_path(self, index: int) -> str:
        df = self.__dataframe__
        ret = df[df["target"] == self.id].reindex().iloc[index]
        return os.path.join(f'fold{ret["fold"]}', ret["filename"])

    def __repr__(self):
        properties = ["name", "label", "length", "childs"]
        return create_repr_str(self, properties)


class US8KDataSource(DataSourceBase):
    def __init__(
        self,
        base_dir: str,
        meta_file: str = "static/meta_file/UrbanSound8K.csv",
        dataframe: pd.DataFrame = None,
        name: str = "US8K",
        label: int = None,
        length: int = None,
    ):
        self.meta_file = meta_file
        childs = None
        if dataframe is None:
            dataframe = pd.read_csv(meta_file)
        self.__dataframe__ = dataframe
        annotations = self.__dataframe__
        annotations["classID"] = annotations["classID"].astype(int)
        annotations = (
            annotations[annotations["classID"] == label]
            if label is not None
            else annotations
        )
        classes = annotations["classID"].unique()
        if len(classes) > 1:
            classes.sort()
            childs = []
            for t in classes:
                tmp = annotations[annotations["classID"] == t]
                childs.append(
                    US8KDataSource(
                        base_dir=base_dir,
                        meta_file=meta_file,
                        dataframe=self.__dataframe__,
                        name=tmp.iloc[0]["class"],
                        length=len(tmp),
                        label=t,
                    )
                )

        super().__init__(
            base_dir=base_dir, name=name, label=label, length=length, childs=childs
        )

    def get_file_path(self, index: int) -> str:
        df = self.__dataframe__
        ret = df[df["classID"] == self.id].reindex().iloc[index]
        return os.path.join(f'fold{ret["fold"]}', ret["slice_file_name"])

    def __repr__(self):
        properties = ["name", "label", "length", "childs"]
        return create_repr_str(self, properties)


class ProvinceDataSource(DataSourceBase):
    def __init__(
        self,
        base_dir: str,
        meta_file: str = "static/meta_file/province.json",
        *,
        name: str,
        label: int = None,
        length: int = None,
        parent: str = None,
    ):
        self.parent = parent
        
        with open(meta_file, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        label_lst = graph.get(name, [])
        childs = None
        if len(label_lst) > 0:
            childs = []
            for i, l in enumerate(label_lst):
                childs.append(
                    ProvinceDataSource(
                        base_dir=base_dir, name=l, label=i, length=1200, parent=name, meta_file=meta_file
                    )
                )
        super().__init__(
            base_dir=base_dir, name=name, label=label, length=length, childs=childs
        )

    def get_file_path(self, index: int) -> str:
        assert self.parent is not None
        assert index < self.length
        return os.path.join(
            self.parent, self.name, f"{self.name}_{index}.wav"
        )

    def __repr__(self):
        properties = ["name", "label", "length", "childs", "parent"]
        return create_repr_str(self, properties)

class BirdclefDataSource(DataSourceBase):
    def __init__(
        self,
        base_dir: str,  # 数据的基目录        
        meta_file: str = "static/meta_file/Birdclef.csv",  # 元数据文件的路径
        name: str = "Birdclef",  # 数据源的名称
        dataframe: pd.DataFrame = None,  # 元数据文件的内容，可以直接提供，也可以通过meta_file参数读取
        label: int = None,  # 用于过滤的标签（类别ID）
        length: int = None,  # 数据源中的数据点数量
    ):
        self.meta_file = meta_file  # 将meta_file参数赋值给实例变量
        childs = None  # 初始化childs变量，用于存储子数据源
        if dataframe is None:
            dataframe = pd.read_csv(self.meta_file)
        self.__dataframe__ = dataframe
        annotations = pd.read_csv(self.meta_file)  # 读取元数据文件
        annotations["id"] = annotations["id"].astype(int)  # 将id列的数据类型设置为int
        annotations = (  # 根据label过滤注释，如果没有提供label，则使用所有注释
            annotations[annotations["id"] == label]
            if label is not None
            else annotations
        )
        classes = annotations["id"].unique()  # 获取所有唯一的类别ID
        if len(classes) > 1:  # 如果有多个类别
            classes.sort()  # 对类别进行排序
            childs = []  # 初始化childs列表
            for t in classes:  # 遍历每个类别
                tmp = annotations[annotations["id"] == t]  # 过滤出当前类别的注释
                childs.append(  # 将新的数据源添加到childs列表
                    BirdclefDataSource(
                        base_dir=base_dir,  # 基目录
                        meta_file=meta_file,  # 元数据文件
                        dataframe=self.__dataframe__,
                        name=tmp.iloc[0]["primary_label"],  # 使用类别的名称
                        length=len(tmp),  # 使用当前类别的数据点数量
                        label=t,  # 使用当前类别的标签
                    )
                )

        super().__init__(  # 调用基类的初始化方法
            base_dir=base_dir,  # 基目录
            name=name,  # 数据源名称
            label=label,  # 标签
            length=length,  # 数据点数量
            childs=childs  # 子数据源
        )

    def get_file_path(self, index: int) -> str:  # 定义一个方法来获取文件路径
        df = self.__dataframe__
        ret = df[df["id"] == self.id].reindex().iloc[index]
        return os.path.join(f'train_audio', ret["filename"]) # 构建并返回文件路径

    def __repr__(self):  # 定义对象的字符串表示
        properties = ["name", "label", "length", "childs"]  # 要包含在表示中的属性列表
        return create_repr_str(self, properties)  # 调用一个辅助函数来创建字符串表示
