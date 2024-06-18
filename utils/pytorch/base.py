import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练器和测试器的基类
class ModelManager:
    def __init__(self, model: torch.nn.Module, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.device = device

    def save_model(self, save_path: str) -> str:
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load_model(self, load_path: str) -> torch.nn.Module:
        self.model.load_state_dict(torch.load(load_path))
        return self.model


