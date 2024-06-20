import gc
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
        self,
        get_file_path: callable = None,
        get_label_name: callable = None,
        tqdm_instance: tqdm = None,
    ) -> tuple[TestMetrics, pd.DataFrame]:
        """
        启动一个测试周期
        """
        self.model.eval()  # 将模型设置为评估模式

        total_loss = 0.0
        bad_cases = []
        outputs_lst = []
        targets_lst = []  # 初始化预测和目标列表

        with torch.no_grad():  # 在此上下文中，所有计算都不会跟踪梯度
            for count, (features, labels, idxes) in enumerate(
                tqdm(self.testing_dataloader, leave=False, desc="[valid]"), start=1
            ):
                # 将数据和标签移动到指定设备（如GPU）
                features = features.to(self.device)
                labels = labels.to(self.device)

                # 使用自动混合精度训练（如果启用）
                if self.using_amp:
                    with amp.autocast(True, dtype=torch.bfloat16):
                        outputs = self.model.forward(features)
                else:
                    outputs = self.model.forward(features)

                outputs = torch.softmax(outputs, dim=1) # softmax
                
                outputs_lst.append(outputs)
                targets_lst.append(labels)
                
                # 计算损失
                loss = self.loss_func(outputs, labels)
                total_loss += loss.item()
                
                pred = outputs.argmax(axis=1)
                results = pred == labels
                # 记录预测错误的结果
                for i, result in enumerate(results):
                    if not result:
                        idx = idxes[i].item()
                        label = labels[i].item()
                        y_pred = pred[i].item()
                        filepath = get_file_path(idx) if get_file_path else None
                        label_name = get_label_name(label) if get_label_name else None
                        prediction = get_label_name(y_pred) if get_label_name else None
                        bad_cases.append({
                            "index": idx,
                            "filepath": filepath,
                            "label": label,
                            "label_name": label_name,
                            "y_pred": y_pred,
                            "prediction": prediction,
                        })

                if tqdm_instance is not None:
                    tqdm_instance.set_description(
                        f"[valid] Progress: {count}/{len(self.testing_dataloader)}"
                    )
                    
        outputs_tensor = torch.cat(outputs_lst, dim=0)
        targets_tensor = torch.cat(targets_lst, dim=0)
        
        metrics_ = self.get_metrics(outputs_tensor, targets_tensor)

        if tqdm_instance is not None:
            tqdm_instance.set_description(
                f"[valid] accuracy: {metrics_.accuracy:.4f}, loss: {metrics_.loss:.4f}"
            )
            
        return metrics_, pd.DataFrame(bad_cases)  # 返回测试指标, 坏样本列表
                    
    def get_metrics(self, outputs_tensor: torch.Tensor, targets_tensor: torch.Tensor)->TestMetrics:
        '''
        获取测试指标
        '''
        
        preds_tensor = torch.argmax(outputs_tensor, dim=1)
        
        # ======================== metrics ======================== #
        metrics_ = TestMetrics()
        
        try:
            metrics_.loss = self.loss_func(outputs_tensor, targets_tensor).item()

            # region 计算auc,prec,rec,acc
            if self.auc_func is not None:
                metrics_.auc = self.auc_func(outputs_tensor, targets_tensor) # 注意这里的pred=outputs_tensor
            if self.prec_func is not None:
                metrics_.precision = self.prec_func(preds_tensor, targets_tensor)
            if self.recall_func is not None:
                metrics_.recall = self.recall_func(preds_tensor, targets_tensor)
            if self.acc_func is not None:
                metrics_.accuracy = self.acc_func(preds_tensor, targets_tensor)
            # endregion

            # region 计算f1_score
            if self.f1_score_func is not None:
                metrics_.f1_score = self.f1_score_func(preds_tensor, targets_tensor)
            if self.f1_score_micro_func is not None:
                metrics_.f1_score_micro = self.f1_score_micro_func(
                    preds_tensor, targets_tensor
                )
            # endregion

            return metrics_
        except Exception as e:
            raise e
        finally:
            del outputs_tensor, targets_tensor, preds_tensor
            gc.collect()

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
