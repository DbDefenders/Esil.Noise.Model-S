# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-06-11 10:49:21
# Description: utils for pytorch

from .trainer import Trainer
from .tester import Tester

__all__ = ["Trainer", "Tester"]

import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import yaml
import json
import wandb
from sklearn import metrics

# from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np
from torchmetrics.functional.classification import (
    auroc,
    precision,
    f1_score,
    recall,
    accuracy,
)
from torch.cuda import amp

from utils import config

"""PyTorch的amp模块可以帮助用户在保持数值稳定性的同时,最大化利用16位浮点数的优势。
自动混合精度训练通常结合使用16位浮点数(float16或半精度)和32位浮点数(float32或全精度)。
16位浮点数可以加快计算速度并减少内存使用,但可能牺牲一些数值稳定性。"""
# region base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FeatureParams = config.features
TagsParams = config.tags
TrainsParams = config.train
# endregion

    
#region init
def initialization(training_dataloader):
    model = load_model(model,TrainsParams['model_path'])
    
    if TrainsParams['optimizer']=='adam':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=TrainsParams['learning_rate'], weight_decay=TrainsParams['weight_decay'])
    else:
        optimizer = load_optimizer(optimizer, TrainsParams['optimizer_path'])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs=TrainsParams["max_epochs"],
        pct_start=0.0,
        steps_per_epoch=len(training_dataloader),
        max_lr=TrainsParams["learning_rate"],
        div_factor=25,
        final_div_factor=4.0e-01,
    )
    
    scaler = amp.GradScaler(enabled=TagsParams['enable_amp'])
    if TrainsParams['loss_type'] == "BCEWithLogitsLoss":
        loss_func = torch.nn.BCEWithLogitsLoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss().to(device)
    return model.to(device), optimizer, scheduler, scaler, loss_func.to(device)
#endregion

#region train
def train_an_epoch(model:torch.nn.Module, optimizer:torch.optim.Optimizer, training_dataloader:torch.utils.data.DataLoader, scheduler,scaler, loss_func,*args):
    trainloss = 0; model.train()  # 初始化训练损失为0，并设置模型为训练模式
    count = 0  # 初始化计数器

    for idx, (features, label) in enumerate(tqdm(training_dataloader,leave=False ,desc="[train]")):  # 遍历数据加载器，使用tqdm显示进度条
        # label = label.reshape(-1, len(LABELS))  # 用于调整标签的形状

        features, label = features.to(device), label.to(
            device
        )  # 将数据和标签移动到指定设备（如GPU）

        optimizer.zero_grad()  # 清空模型的梯度
        with amp.autocast(TagsParams['enable_amp'], dtype=torch.bfloat16):  # 决定是否启用自动混合精度训练，并设置数据类型为torch.bfloat16
            pred = model.forward(features)  # 通过模型前向传播得到预测结果
            loss = loss_func(pred, label)  # 计算损失

        scaler.scale(loss).backward()  # 使用梯度缩放器缩放损失并计算梯度,反向传播
        scaler.step(optimizer)  # 更新模型的参数
        scaler.update()  # 更新梯度缩放器的状态

        scheduler.step()  # 更新学习率调度器的状态

        trainloss += loss.item()  # 累加训练损失
        del features, label, loss  # 手动删除变量以释放内存
        count += 1  # 计数器加1
        # if count == 300:
        # break  # 注释掉的代码，可能用于限制训练的批量数
        
    trainloss /= len(training_dataloader)  # 计算平均训练损失
    '''计算累加训练损失：在训练过程中，对于每个批次，我们都会计算模型的损失。通过累加这些损失，我们可以得到整个训练集的总损失。
    在训练周期结束时，我们将总损失除以训练集中的批次数量，得到平均损失。
    这个平均损失是评估模型在训练集上表现的一个重要指标。通过监控平均损失的变化，我们可以了解模型是否正在学习以及学习的效果如何。'''
    if TagsParams['wandb'] == True:  # 使用Weights & Biases记录训练损失和学习率
        wandb.log({f"train_loss": trainloss, f"lr":scheduler.get_lr()[0]})
    return model, optimizer, scaler, scheduler, trainloss  # 返回更新后的模型、优化器、梯度缩放器、学习率调度器和平均训练损失

#endregion
#region test
def test_an_epoch(model, training_dataloader, loss_func, LABELS:list, *args):
    validloss = 0  # 初始化验证集损失
    model.eval()  # 将模型设置为评估模式
    preds, targets = [], []  # 初始化预测和目标列表
    with torch.no_grad():  # 在此上下文中，所有计算都不会跟踪梯度
        for idx, (features, label) in enumerate(tqdm(training_dataloader, leave=False, desc="[valid]")):
            # label = label.reshape(-1, len(LABELS))  # 这一行被注释掉了，如果需要调整标签的形状，应该取消注释

            features ,label = features.to(device), label.to(device)  # 将数据和标签移动到相应的设备上
            with amp.autocast(TagsParams['enable_amp'], dtype=torch.bfloat16): 
                # 使用自动混合精度训练（如果启用）
                pred = model.forward(features)  # 通过模型前向传播
                preds.extend(pred.detach().numpy())  # 将预测结果添加到列表中
                targets.extend(label.numpy())  # 将目标标签添加到列表中

    # ======================== metrics ========================#
    # 将列表转换为张量
    preds_tensor = torch.tensor(np.array(preds))
    targets_tensor = torch.tensor(np.array(targets))
    preds_argmax = preds_tensor.argmax(axis=1)  # 获取每个预测的最大概率索引

    # 计算loss
    validloss = loss_func(preds_tensor, targets_tensor)  # 计算验证集损失
    if isinstance(validloss, torch.Tensor):
        validloss = validloss.item()
    # 计算auc,prec,rec,acc
    auc = auroc(
        preds=preds_tensor,
        task="multiclass",
        target=targets_tensor,
        num_classes=len(LABELS),
        average="macro",
    )  # 计算AUC

    prec = precision(
        preds=preds_tensor,
        task="multiclass",
        target=targets_tensor,
        num_classes=len(LABELS),
        average="macro",
    )  # 计算精确度
    rec = recall(
        preds=preds_tensor,
        task="multiclass",
        target=targets_tensor,
        num_classes=len(LABELS),
        average="macro",
    )  # 计算召回率

    acc = accuracy(
        preds=preds_tensor,
        task="multiclass",
        target=targets_tensor,
        num_classes=len(LABELS),
        average="macro",
    )  # 计算准确率

    # 计算f1
    # sk_f1 = metrics.f1_score(np.array(y_trues), np.array(targets), average="micro")  #参数是反着的
    f1 = f1_score(preds=preds_argmax,task='multiclass', target=targets_tensor, num_classes=len(LABELS),
                             average="macro")  # 计算F1分数（宏观平均）
    f1_micro = f1_score(preds=preds_argmax,task='multiclass', target=targets_tensor, num_classes=len(LABELS),
                            average="micro")  # 计算F1分数（微观平均）

    if TagsParams['wandb']  == True:
        wandb.log({f"valid_loss": validloss,
                   f"AUC":auc,
                   f"precision":prec, 
                   f"recall":rec, 
                   f"accuracy":acc,
                   f"F1":f1,
                   f"F1_micro":f1_micro})
                  
    return validloss, auc, prec, rec, acc, f1, f1_micro  # 返回验证集损失和评估指标


# endregion


# region save & load
def save_model(model, save_path, *args):
    torch.save(model.state_dict(), save_path)


def save_optimizer(optimizer, save_path, *args):
    torch.save(optimizer.state_dict(), save_path)


def save_latest_epoch_info(file, data: dict, *args):
    with open(file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def load_model(model, model_path, *args):
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    return model


def load_optimizer(optimizer, optimizer_path):
    optimizer_state_dict = torch.load(optimizer_path)
    optimizer.load_state_dict(optimizer_state_dict)
    return optimizer


def load_latest_epoch_info(file, *args):
    with open(file, "r", encoding="utf-8") as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
        return info


# endregion
