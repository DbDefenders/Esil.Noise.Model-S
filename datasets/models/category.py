from typing import Union
from collections import Counter
from copy import deepcopy

from sklearn.model_selection import train_test_split

from .base import DataSourceBase
from .label import Label
from utils.common import BisectList, get_child
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "static/fonts/SimHei.ttf"  # 替换为 SimHei 字体的实际路径
fm.fontManager.addfont(font_path)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置默认字体
plt.rcParams["axes.unicode_minus"] = False


class Category(BisectList):
    def __init__(self, name: str, labels: list[Union[Label, DataSourceBase]]):
        labels = deepcopy(labels)
        for i, l in enumerate(labels):
            if isinstance(l, DataSourceBase):
                labels[i] = l.to_label()
        self.name = name
        super().__init__(lst_of_lst=labels, sort_key="name", reset_index="id")

    @property
    def labels(self) -> list[Label]:
        return self._lst_of_lst

    @labels.setter
    def labels(self, new_labels: list[Label]):
        raise NotImplementedError("Category.labels is a read-only property.")

    @property
    def labels_info(self) -> list:
        """标签信息"""
        return [{"id": l.id, "name": l.name, "length": len(l), "sources": ",".join([repr(s) for s in l.sources])} for l in self.labels]

    @property
    def labels_dataframe(self) -> pd.DataFrame:
        """标签信息"""
        return pd.DataFrame(self.labels_info)

    def labels_barplt(self, legend_tag: bool = False) -> plt.Figure:
        """标签柱状图"""
        # 准备数据
        ids, lengths, names = zip(*[(l.id, len(l), l.name) for l in self.labels])

        # 创建柱状图
        fig, ax = plt.subplots(
            figsize=(20, 12)
        )  # 使用 subplots() 获取 figure 和 axis 对象
        bars = ax.bar(ids, lengths)  # 使用标签 id 作为 x 轴

        # 设置标题和轴标签
        ax.set_title("Labels Info")
        ax.set_xlabel("Label ID")
        ax.set_ylabel("Frequency")

        if legend_tag:
            # 为每个条形添加图例标签
            ax.bar_label(bars, labels=names, label_type="edge")
            # 添加图例
            ax.legend(bars, names)

        # 返回 figure 对象，以便于进一步操作
        return fig

    def labels_scatterplt(self, legend_tag: bool = False) -> plt.Figure:
        ids, lengths, names = zip(*[(l.id, len(l), l.name) for l in self.labels])

        # 创建散点图
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(ids, lengths)

        # 设置标题和轴标签
        ax.set_title("Labels Info Scatter")
        ax.set_xlabel("Label ID")
        ax.set_ylabel("Frequency")
        if legend_tag:
            # 为每个点添加标签（如果数据点不是很多的话）
            for i, txt in enumerate(names):
                ax.annotate(txt, (ids[i], lengths[i]))

        return fig

    def get_label(self, index):
        return get_child(index, self.labels)

    def __repr__(self):
        return f"Category(name='{self.name}', labels={self._lst_of_lst})"

    def __getitem__(self, index):
        label_idx, _ = self.get_next_node_idx(index)
        label = self.labels[label_idx]
        return self.find_in_next_node(index), label.id

    def get_train_test_data(self, test_ratio: float = 0.2, seed: int = 1202):
        """
        获取训练集和测试集数据
        :param test_ratio: 测试集比例
        :param seed: 随机种子
        :return: X_train, X_test, y_train, y_test
        """
        X, y = zip(*([i for i in self]))

        # 按比例分割数据，并且分层
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=seed
        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def count_train_test_data(*, y_train, y_test):
        """
        计数训练集和测试集的分类分布

        :param y_train: 训练集标签
        :param y_test: 测试集标签

        :return: counter_train, counter_test
        """
        counter_train = Counter(y_train)
        counter_test = Counter(y_test)
        print("训练集分类分布:", counter_train)
        print("测试集分类分布:", counter_test)
        return counter_train, counter_test
