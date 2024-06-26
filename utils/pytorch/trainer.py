import gc
from .base import ModelManager, DEVICE
import torch
from torch.cuda import amp
from tqdm import tqdm

class Trainer(ModelManager):
    def __init__(
        self,
        model: torch.nn.Module,
        training_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        using_amp: bool = False,
        device: torch.device = DEVICE,
    ):
        """
        初始化训练器
        :param epochs: 训练的轮数
        :param model: 待训练的模型
        :param training_dataloader: 训练数据集的DataLoader
        :param loss_func: 损失函数
        :param optimizer: 优化器
        :param scheduler: 学习率调度器
        :param using_amp: 是否使用自动混合精度训练
        """
        super().__init__(model, device)

        self.training_dataloader = training_dataloader
        self.loss_func = loss_func.to(device)

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                params=model.parameters(), lr=1e-3, weight_decay=1e-5
            )
        self.optimizer = optimizer
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer, step_size=10, gamma=0.1
            )
        self.scheduler = scheduler

        if using_amp:
            self.scaler = amp.GradScaler(enabled=True)

        else:
            self.scaler = None

    def save_optimizer(self, save_path)->str:
        torch.save(self.optimizer.state_dict(), save_path)
        return save_path
        
    def reload_optimizer(self, optimizer_path, scheduler: torch.optim.lr_scheduler.LRScheduler = None)->'Trainer':
        optimizer_state_dict = torch.load(optimizer_path)
        self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer, step_size=10, gamma=0.1
            )
        self.scheduler = scheduler
        return self
    
    def reload_trainer(self, model_path, optimizer_path)->'Trainer':
        self.load_model(model_path)
        self.reload_optimizer(optimizer_path)
        return self

    def train_an_epoch(self, tqdm_instance:tqdm=None, use_inner_tqdm:bool=True)->float:
        """
        启动一个训练周期
        """
        training_loss = 0
        self.model.train()

        for count, (features, labels, _) in enumerate(
            tqdm(self.training_dataloader, leave=False, desc="[train]") if use_inner_tqdm else self.training_dataloader,
            start=1,  # 遍历数据加载器，使用tqdm显示进度条
        ):
            # 将数据和标签移动到指定设备（如GPU）
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()  # 清空模型的梯度

            if self.scaler is not None:  # 启用了自动混合精度训练
                with amp.autocast(enabled=True):
                    pred = self.model.forward(features)  # 通过模型前向传播得到预测结果
                    pred = torch.softmax(pred, dim=1) # softmax
                    loss = self.loss_func(pred, labels)  # 计算损失
                self.scaler.scale(
                    loss
                ).backward()  # 使用梯度缩放器缩放损失并计算梯度,反向传播
                self.scaler.step(self.optimizer)  # 更新模型的参数
                self.scaler.update()  # 更新梯度缩放器的状态
            else:  # 没有启用自动混合精度训练
                pred = self.model.forward(features)  # 通过模型前向传播得到预测结果
                pred = torch.softmax(pred, dim=1) # softmax
                loss = self.loss_func(pred, labels)  # 计算损失
                loss.backward()  # 反向传播
            self.optimizer.step()  # 更新模型的参数
            self.scheduler.step()  # 更新学习率调度器的状态

            training_loss += loss.item()  # 累加训练损失

            del features, labels, pred, loss  # 手动删除变量以释放内存
            gc.collect()  # 手动调用垃圾回收器以释放内存
            if tqdm_instance is not None:
                tqdm_instance.set_description(f"[train] loss: {training_loss/count:.4f} Progress: {count}/{len(self.training_dataloader)}")

        training_loss /= len(self.training_dataloader)  # 计算平均训练损失
        return training_loss
