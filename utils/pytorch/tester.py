import torch
import pandas as pd
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
from pydantic import BaseModel, Field
import torchmetrics.functional.classification as tmf

from .base import ModelManager, DEVICE
from .trainer import Trainer
from ..decorators import tensor_to_number


class TestMetrics(BaseModel):
    # 测试指标
    loss: float = Field(default=None, fdescription="loss")
    auc: float = Field(default=None, description="AUC")
    accuracy: float = Field(default=None, description="Accuracy")
    precision: float = Field(default=None, description="Precision")
    recall: float = Field(default=None, description="Recall")
    f1_score: float = Field(default=None, description="F1_score_marco")
    f1_score_micro: float = Field(default=None, description="F1_score_micro")


class Tester(ModelManager):
    def __init__(
        self,
        model: torch.nn.Module,
        testing_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        num_classes: int,
        using_amp: bool = False,
        acc_func: callable = tmf.accuracy,
        prec_func: callable = tmf.precision,
        auc_func: callable = tmf.auroc,
        recall_func: callable = tmf.recall,
        f1_score_func: callable = tmf.f1_score,
        device: torch.device = DEVICE,
    ):
        """
        初始化测试器
        :param model: 待测试的模型
        :param testing_dataloader: 测试数据集的DataLoader
        :param loss_func: 损失函数
        :param label_lst: 标签列表
        :param using_amp: 是否使用自动混合精度训练
        :param acc_func: 准确率函数
        :param prec_func: 精确率函数
        :param auc_func: auc函数
        :param recall_func: 召回率函数
        :param f1_score_func: f1_score函数
        :param device: 运行设备
        """
        # region init
        super().__init__(model, device)

        self.testing_dataloader = testing_dataloader
        self.loss_func = loss_func.to(device)
        self.num_classes = num_classes
        self.using_amp = using_amp
        self.acc_func = self.get_metrics_func(acc_func)
        self.auc_func = self.get_metrics_func(auc_func)
        self.prec_func = self.get_metrics_func(prec_func)
        self.recall_func = self.get_metrics_func(recall_func)
        self.f1_score_func = self.get_metrics_func(f1_score_func, average="macro")
        self.f1_score_micro_func = self.get_metrics_func(f1_score_func, average="micro")

        self.best_accuracy = 0.0
        # endregion

    def set_best_accuracy(self, accuracy: float) -> "Tester":
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        return self

    def get_metrics_func(self, func: callable, average: str = "macro") -> callable:
        """
        获取指标函数

        :param func: 指标函数
        :param num_classes: 类别数
        :param average: 平均方式
        """
        if func is None:
            return None
        else:
            metrics_func = lambda preds, targets: (
                func(
                    preds,
                    targets,
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=average,
                )
            )
            return tensor_to_number(metrics_func)

    def test_an_epoch(
        self, get_file_path: callable = None, get_label_name:callable = None, tqdm_instance: tqdm = None
    ) -> tuple[TestMetrics, pd.DataFrame]:
        """
        启动一个测试周期
        """
        self.model.eval()  # 将模型设置为评估模式

        columns = [
            "index",
            "Filename",
            "label",
            "y_pred",
        ]  # idx, filepath, label, y_pred

        bad_cases = []
        preds, targets = [], []  # 初始化预测和目标列表
        with torch.no_grad():  # 在此上下文中，所有计算都不会跟踪梯度
            for count, (features, labels, idxes) in enumerate(
                tqdm(self.testing_dataloader, leave=False, desc="[valid]"), start=1
            ):
                # 将数据和标签移动到指定设备（如GPU）
                features = features.to(self.device)
                labels = labels.to(self.device)
                if self.using_amp:  # 使用自动混合精度训练（如果启用）
                    with amp.autocast(True, dtype=torch.bfloat16):
                        pred = self.model.forward(features)  # 通过模型前向传播
                        preds.extend(pred.float().cpu().detach().numpy())
                        targets.extend(labels.cpu().numpy())
                else:  # 没有启用自动混合精度训练
                    pred = self.model.forward(features)  # 通过模型前向传播
                    preds.extend(pred.float().cpu().detach().numpy())
                    targets.extend(labels.cpu().numpy())

                outputs = torch.softmax(pred, dim=1)
                results = outputs.argmax(axis=1) == labels

                # 使用wandb记录测试结果
                for i, result in enumerate(results):
                    label = labels[i].item()
                    y_pred = outputs.argmax(1)[i].cpu().numpy().item()
                    idx = idxes[i].item()

                    if get_file_path is not None:
                        filepath = get_file_path(idx)
                    else:
                        filepath = None
                        
                    if get_label_name is not None:
                        label_name = get_label_name(label)
                        prediction = get_label_name(y_pred)
                    else:
                        label_name = None
                        prediction = None

                    if not result:
                        bad_cases.append(
                            {
                                "index": idx,
                                "filepath": filepath,
                                "label": label,
                                "label_name": label_name,
                                "y_pred": y_pred,
                                "prediction": prediction
                            }
                        )

                if tqdm_instance is not None:
                    tqdm_instance.set_description(f"[valid] Progress: {count}/{len(self.testing_dataloader)}")
        preds_tensor = torch.tensor(np.array(preds))
        targets_tensor = torch.tensor(np.array(targets))
        preds_argmax = preds_tensor.argmax(axis=1)  # 获取每个预测的最大概率索引

        # ======================== metrics ======================== #
        metrics_ = TestMetrics()

        # region 计算loss
        valid_loss = self.loss_func(preds_tensor, targets_tensor)  # 计算验证集损失
        if isinstance(valid_loss, torch.Tensor):
            valid_loss = valid_loss.item()
        metrics_.loss = valid_loss
        # endregion

        # region 计算auc,prec,rec,acc
        if self.auc_func is not None:
            auc = self.auc_func(preds_tensor, targets_tensor)
            metrics_.auc = auc.item() if isinstance(auc, torch.Tensor) else auc
        if self.prec_func is not None:
            precision = self.prec_func(preds_tensor, targets_tensor)
            metrics_.precision = (
                precision.item() if isinstance(precision, torch.Tensor) else precision
            )
        if self.recall_func is not None:
            recall = self.recall_func(preds_tensor, targets_tensor)
            metrics_.recall = (
                recall.item() if isinstance(recall, torch.Tensor) else recall
            )
        if self.acc_func is not None:
            accuracy = self.acc_func(preds_tensor, targets_tensor)
            metrics_.accuracy = (
                accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
            )
        # endregion

        # region 计算f1_score
        if self.f1_score_func is not None:
            f1_score = self.f1_score_func(preds_argmax, targets_tensor)

        if self.f1_score_micro_func is not None:
            metrics_.f1_score_micro = self.f1_score_micro_func(
                preds_argmax, targets_tensor
            )
        # endregion
        
        if tqdm_instance is not None:
            tqdm_instance.set_description(f"[valid] accuracy: {metrics_.accuracy:.4f}, loss: {metrics_.loss:.4f}")
        return metrics_, pd.DataFrame(bad_cases)  # 返回测试指标

        # ret = metrics_.model_dump()  # 序列化测试指标

        # return {k: v for k, v in ret.items() if v is not None}

    @classmethod
    def from_trainer(
        cls,
        trainer: Trainer,
        testing_dataloader: torch.utils.data.DataLoader,
        num_classes: int,
        accuracy: float,
        acc_func: callable = tmf.accuracy,
        prec_func: callable = tmf.precision,
        auc_func: callable = tmf.auroc,
        recall_func: callable = tmf.recall,
        f1_score_func: callable = tmf.f1_score,
    ):
        """
        从训练器中创建测试器

        :param trainer: 训练器
        :param testing_dataloader: 测试数据集的DataLoader
        :param label_lst: 标签列表
        """
        ret = cls(
            model=trainer.model,
            testing_dataloader=testing_dataloader,
            loss_func=trainer.loss_func,
            num_classes=num_classes,
            using_amp=trainer.scaler is not None,
            acc_func=acc_func,
            prec_func=prec_func,
            auc_func=auc_func,
            recall_func=recall_func,
            f1_score_func=f1_score_func,
            device=trainer.device,
        )

        ret.set_best_accuracy(accuracy)
        return ret
