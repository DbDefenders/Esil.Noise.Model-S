# Get Start!

> Recommended: python>=3.10

## Install requirements

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Install ffmpeg

### Linux

1. 使用包管理器安装 `ffmpeg`。例如，在 Ubuntu 上，你可以运行以下命令：
   ```bash
   sudo apt-get install ffmpeg
   ```
2. 确保 `ffmpeg` 和 `ffprobe` 已经添加到系统路径中。你可以通过运行以下命令来检查：
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

### macOS

1. 使用 Homebrew 安装 `ffmpeg`。运行以下命令：
   ```bash
   brew install ffmpeg
   ```
2. 确保 `ffmpeg` 和 `ffprobe` 已经添加到系统路径中。你可以通过运行以下命令来检查：
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

### Windows

> 参考链接：[Download FFmpeg](https://ffmpeg.org/download.html#build-windows)

1. 下载 `ffmpeg` 的 Windows 版本并解压缩。
2. 将 `ffmpeg` 和 `ffprobe` 的路径添加到系统的环境变量中。

## Set Environment Variable and Configs

### Copy a `.env` file from `.env.example`

#### Windows cmd

```shell
copy .env.example .env
```

#### Linux

```shell
cp .env.example .env
```

### Fill environment variable in `.env`

- `WANDB_API_KEY` [W&amp;A-QuickStart](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart)
- `GRADIO_ROOT_PATH` root path of gradio app, defaults to ""
- `GRADIO_SERVER_PORT` server port of gradio app, defaults to "8080"

### Copy a `configs.yml` file from `configs.yml.example`

#### Windows cmd

```shell
copy configs.yml.example configs.yml
```

#### Linux

```shell
cp configs.yml.example configs.yml
```

# datasets

- `DataSource` 数据源：目前支持的数据集（datasets.SupportedSources）包括ESC50，US8K，以及五大类噪声。
- `Label` 自定义数据标签
- `Category` 数据类别
- `DatasetFactory` 根据 `Category`创建数据集的工厂

## 添加新的数据源（数据集）

1、在 `datasets/models/sources.py`中定义新的数据源类型

```python
class CustomDatasource(DataSourceBase):
    def __init__(self, base_dir: str, name: str, label: int=None, childs: int = None, **other_kwargs):
  	# set other kwargs
	# self.xxx = xxx
        childs = None
        # 根据数据集添加子类别
        if has_childs:
            childs = []
            for i in range(num_childs):
                childs.append(CustomDatasource(base_dir, name, i))
        super().__init__(base_dir=base_dir, name=name, label=label, childs=childs)

    def get_file_path(self, index: int) -> str:
        raise NotImplementedError("需要实现get_file_path方法")
  
    def __repr__(self):
        properties = ["base_dir", "name", "label", "childs"] # and any other properties you want to include
        return create_repr_str(self, properties)
```

2、在 `configs.yml`中的 `DataSources`的添加新的数据源的参数，例如： `base_dir`和 `meta_file`等

3、在 `datasets/SupportedSources.py`中添加新的数据源实例

```python
CUSTOM = CustomDatasource(**datasources_info['Custom'], **other_kwargs)

# __all__.append("CUSTOM")
```

## 数据集的创建流程

0、检查 `configs.yml`中的 `DataSources`的 `base_dir`和 `meta_file`路径是否正确

1、根据  `SupportedSources` 创建自定义数据标签列表 `labels`

2、根据数据标签列表 `labels` 创建数据类别 `category`

3、使用  `category` 创建 `DatasetFactory`

4、调用 `DatasetFactory` 中的 `create_dataset` 创建训练集或测试集

> sources -> labels -> category -> dataset_factory -> train_data, test_data

> 参考: "test_dataset_factory.ipynb"

# Documents of utils

## wlog

### class LogFileType : wandb files type

### class WandbLogger: logger with wandb

## pytorch

### initialization 初始化模型、优化器、缩放器、损失器

# Todo

* [X] 自由搭配数据集
* [X] 模型训练与测试
* [ ] 音频分析
