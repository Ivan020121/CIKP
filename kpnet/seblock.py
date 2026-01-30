
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    通道注意力模块 (Squeeze-and-Excitation, 适用于输入张量形状 (N, C))
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (N, C)
        # Squeeze：对batch内每个样本的特征取全局平均（这里可以直接复用x本身）
        y = x.mean(dim=0, keepdim=True)  # (1, C)
        # Excitation：生成通道权重
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)             # (1, C)
        # Reweight：按通道加权
        x = x * y                       # 广播到 (N, C)
        return x