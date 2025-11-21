from typing import Tuple
import torch.nn as nn
from torchvision import models

_RESNET_FACTORY = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

class ResNetEncoder(nn.Module):
    """
    Backbone wrapper that exposes feature vectors without a classifier head.
    Adapts structure based on dataset resolution.
    """

    def __init__(self, backbone_name: str = "resnet18", weights=None, dataset: str = "cifar10"):
        """
        Args:
            backbone_name: 'resnet18', 'resnet34', etc.
            weights: Pretrained weights (e.g., models.ResNet18_Weights.IMAGENET1K_V1)
            dataset: 'cifar10' or 'stl10' (controls the stem structure)
        """
        super().__init__()
        backbone_name = backbone_name.lower()
        dataset = dataset.lower()
        
        if backbone_name not in _RESNET_FACTORY:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")

        backbone_fn = _RESNET_FACTORY[backbone_name]
        
        # 1. 加载骨干网络 (如果传入了 weights，这里会加载 ImageNet 权重)
        backbone = backbone_fn(weights=weights)
        
        # 2. 根据数据集修改第一层结构 (Stem)
        if "cifar" in dataset:
            print(f"-> Modifying {backbone_name} stem for CIFAR (32x32)...")
            # CIFAR: 3x3 Conv, Stride 1, No MaxPool
            # 注意：这将丢弃预训练模型中 conv1 的权重（如果使用了预训练）
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            
            # 如果使用了预训练权重，新初始化的 conv1 需要很好的初始化
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            
        elif "stl" in dataset:
            print(f"-> Using standard {backbone_name} stem for STL-10 (96x96)...")
            # STL-10: 96x96 足够大，可以使用标准结构 (7x7 Conv stride 2 + MaxPool)
            # 这样可以保留预训练的 conv1 权重，且显存占用更小。
            # 也可以选择只去掉 maxpool (视具体需求而定)，但标准结构通常能跑出不错的结果。
            pass 
            
        else:
            # 默认为 ImageNet 标准结构
            pass

        # 3. 去掉全连接层 (FC Head)
        self.out_dim: int = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    @property
    def feature_dim(self) -> int:
        return self.out_dim