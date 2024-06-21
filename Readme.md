# Get Start!

> IDE: vs code，选择System Installer版本
> Python环境管理：miniconda或anaconda，python版本使用3.10
> Linux: Xshell，Xftp

## Install requirements

`requirements.txt`文件是一个列出项目依赖的文本文件。

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Install ffmpeg

`ffmpeg` 是一个非常强大的开源工具，用于处理视频和音频文件。

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

`.env`文件是一个用于存储环境变量(数据库凭据、API密钥、密码)的文件，通常用于配置应用程序的运行环境。

`.env.example`文件是一个示例环境变量文件，用于指导规范开发者定义变量名以及设置值。

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

`configs.yml`是一个配置文件，通常用于存储应用程序的配置信息。

`configs.yml.example` 文件是一个示例环境变量文件，用于指导规范开发者定义变量名以及设置值。

#### Windows cmd

```shell
copy configs.yml.example configs.yml
```

#### Linux

```shell
cp configs.yml.example configs.yml
```

# Datasets

- `DataSource` 数据源：目前支持的数据集（datasets.SupportedSources）包括ESC50，US8K，BIRDCLEF，TRAFFIC，NATURE，INDUSTRIAL，SOCIAL，CONSTRUCTIONAL，etc..
- `Label` 数据标签：自定义数据标签，etc..
- `Category` 数据类别：展示标签信息，绘制柱状、散点图，获取训练集和测试集数据，计数训练集和测试集的分类分布，etc..
- `DatasetFactory` 根据 `Category`创建数据集的工厂：计数训练集和测试集样本数，创建训练集或测试集数据集，etc..

## 添加新的数据源（数据集）

1、在 `datasets/models/sources.py`中定义新的数据源类型

```python
class CustomDatasource(DataSourceBase):
    def __init__(self, base_dir: str, meta_file: str, dataframe: pd.DataFrame, name: str, label: int=None, length: int, **other_kwargs):
  	# set other kwargs
	# self.xxx = xxx
        childs = None
	#……创建子类别
        super().__init__(base_dir=base_dir, name=name, label=label, length=length, childs=childs)

    def get_file_path(self, index: int) -> str:
        df = self.__dataframe__
        ret = df[df["target"] == self.id].reindex().iloc[index]
        return os.path.join(f'fold{ret["fold"]}', ret["filename"])
  
    def __repr__(self):
        properties = ["base_dir", "name", "label", "childs"] # and any other properties you want to include
        return create_repr_str(self, properties)
```

2、在 `configs.yml`中的 `DataSources`的添加新的数据源的参数，例如： `base_dir`和 `meta_file`等

```
DataSources:
# 数据源，linux下路径中要用“/”,不能用“\”
  CUSTOM:
    base_dir: c:\custom
    meta_file: static/meta_file/CUSTOM.csv
```

3、在 `datasets/SupportedSources.py`中添加新的数据源实例

```python
from .models.sources import CustomDataSource
class SupportedSourceTypes(Enum):
    CUSTOM = {"class": CustomDataSource, "args":{**datasources_info["CUSTOM"]}}
```

## 数据集的创建流程

0、检查 `configs.yml`中的 `DataSources`的 `base_dir`和 `meta_file`路径是否正确

1、根据  `SupportedSources` 创建自定义数据标签列表 `labels`

2、根据数据标签列表 `labels` 创建数据类别 `category`

3、使用  `category` 创建 `DatasetFactory`

4、调用 `DatasetFactory` 中的 `create_dataset` 创建训练集或测试集

> sources -> labels -> category -> dataset_factory -> train_data, test_data

> 参考: "03exp_dataset_factory.ipynb"

# Models

`model`模型：目前收集的模型以及模型结构，包括CNN，RESNET，PANN，VGG，etc..

# Static

`docs`:超参记录文档、笔记等

`fonts`：代码使用字体等

`meta_file`：数据集的meta文件，包括数据音频路径，标签，时间等

# Documents of utils

`_Config`：基于 `config.yml`初始化参数

## audio

`extractor`：定义一个音频特征提取器类，包括spectrogram，mel_spec，mfcc，lfcc，features_map，etc..

`feature`: 基于`extractor`提取特征

`process`：音频处理方法，包括重采样，音频信号声道融合，裁剪音频信号到指定长度，填充音频信号到指定长度

## decorators

`tensor_to_number`：将函数的输出从 `torch.Tensor`对象转换为数值

## math

`normalization`：归一化方法，包括min_max_normalize，z_scroe_normailze

`relu`：激活函数，包括limit_relu，percent_relu

## plot

`get_spectrogram`：获取频谱特征

`get_mel_spectrogram`：获取梅尔频谱特征

`plot_frequency_spectrogram`：绘制频率谱图

`plot_waveform`：绘制波形图

`plot_spectrogram`：绘制频谱图

`plot_fbank`：绘制滤波器组图像

## pytorch

`trainer`：初始化训练器，进行训练。

### 训练步骤

1. 导入训练依赖以及环境变量
2. 定义超参数，包括轮数，采样率，片段时长，批次大小，精度阈值，特征提取方法等
3. 定义特征提取器，数据加载器，模型
4. 初始化wandb，定义实验训练存档
5. 定义实验category
6. 创建数据集工厂对象，并创建训练集和测试集数据集对象
7. 定义函数，用于获取测试集的音频文件名和标签名称，用于在测试时记录预测错误的结果
8. 初始化训练器，包括dataloader，model，loss，trainer
9. 开始训练
10. 使用wandb记录训练、验证结果

`tester`：初始化测试器，进行测试。

## wlog

`plt2image`：将matplotlib的 `Figure`对象转换为Weights & Biases（wandb）日志记录系统可识别的 `Image`对象

`df2table`：将Pandas的 `DataFrame`对象转换为Weights & Biases（wandb）日志记录系统可识别的 `Table`对象。

> 模型训练参考："06exp_trainer_tester.ipynb"

# Todo

* [X] 自由搭配数据集
* [X] 模型训练与测试
* [ ] 音频分析
* [ ] wandb sweep调参