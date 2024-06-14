import os
import json
import pandas as pd
from utils.common import create_repr_str
from .base import DataSourceBase

class ESC50DataSource(DataSourceBase):
    def __init__(
        self,
        meta_file,
        base_dir,
        name: str = "ESC50",
        label: int = None,
        length: int = None,
    ):
        self.meta_file = meta_file

        childs = None
        annotations = pd.read_csv(meta_file)
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
                        name=tmp.iloc[0]["category"],
                        label=t,
                        length=len(tmp),
                    )
                )

        super().__init__(
            base_dir=base_dir, name=name, label=label, length=length, childs=childs
        )

    def get_file_path(self, index) -> str:
        df = pd.read_csv(self.meta_file)
        ret = df[df["target"] == self.id].reindex().iloc[index]
        return os.path.join(f'fold{ret["fold"]}', ret["filename"])

    def __repr__(self):
        properties = ["meta_file", "base_dir", "name", "label", "length", "childs"]
        return create_repr_str(self, properties)


class US8KDataSource(DataSourceBase):
    def __init__(
        self,
        meta_file,
        base_dir,
        name: str = "US8K",
        label: int = None,
        length: int = None,
    ):
        self.meta_file = meta_file

        childs = None
        annotations = pd.read_csv(meta_file)
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
                        name=tmp.iloc[0]["class"],
                        length=len(tmp),
                        label=t,
                    )
                )

        super().__init__(
            base_dir=base_dir, name=name, label=label, length=length, childs=childs
        )

    def get_file_path(self, index) -> str:
        df = pd.read_csv(self.meta_file)
        ret = df[df["classID"] == self.id].reindex().iloc[index]
        return os.path.join(f'fold{ret["fold"]}', ret["slice_file_name"])

    def __repr__(self):
        properties = ["meta_file", "base_dir", "name", "label", "length", "childs"]
        return create_repr_str(self, properties)


class ProvinceDataSource(DataSourceBase):
    def __init__(
        self,
        base_dir: str,
        meta_file: str,
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

    def get_file_path(self, index) -> str:
        assert self.parent is not None
        assert index < self.length
        return os.path.join(
            self.base_dir, self.parent, self.name, f"{self.name}_{index}.wav"
        )

    def __repr__(self):
        properties = ["base_dir", "name", "label", "length", "childs", "parent"]
        return create_repr_str(self, properties)