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

1. 下载 `ffmpeg` 的 Windows 版本并解压缩。
2. 将 `ffmpeg` 和 `ffprobe` 的路径添加到系统的环境变量中。

## Set Environment Variable

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

# Documents of utils

## wlog

### class LogFileType : wandb files type

### class WandbLogger

    `init_wandb`: 初始化wandb

## pytorch
### initialization 初始化  model, optimizer, scheduler, scaler, loss_func

### train_an_epoch 构建一个epoch的训练流程

### test_an_epoch 构建一个epoch的测试和验证流程

### save_model 保存模型
### save_optimizer 保存优化器
### save_latest_epoch_info 保存最近epoch信息
### load_model 加载模型
### load_optimizer 加载优化器
### load_latest_epoch_info 加载最近epoch信息




# Todo
