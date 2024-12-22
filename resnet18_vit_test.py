import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
# 定义ResNet-18作为特征提取器
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # 移除ResNet-18的全连接层

# 定义ViT作为特征转换器和分类器
from torchvision.transforms import transforms
import timm

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

vit = timm.create_model('vit_base_patch16_224', pretrained=True)
print(vit)

# 构建级联模型
class CascadeModel(nn.Module):
    def __init__(self, resnet, vit):
        super(CascadeModel, self).__init__()
        self.resnet = resnet
        self.vit = vit

    def forward(self, x):
        features = self.resnet(x)  # ResNet-18特征提取
        features = features.view(features.size(0), -1)  # 将特征展平
        features = features.unsqueeze(-1).unsqueeze(-1)
        print(np.shape(features))
        features = F.interpolate(features, size=(384, 384), mode='bilinear', align_corners=False)
        output = self.vit(features)  # ViT特征转换和分类
        return output

# 创建级联模型实例
cascade_model = CascadeModel(resnet, vit)
import torch

# 将cascade_model移动到CUDA设备上
cascade_model = cascade_model.cuda()


from torch.utils.data import DataLoader

# 假设有训练集和测试集的数据集 train_dataset 和 test_dataset

# 定义训练集和测试集的数据加载器
batch_size = 32

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# 定义训练集和测试集的数据集对象
train_dataset = CIFAR10(root='path_to_data', train=True, download=True, transform=ToTensor())
test_dataset = CIFAR10(root='path_to_data', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 使用级联模型进行训练和测试
...
# 假设有训练集和测试集的数据加载器 train_loader 和 test_loader

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cascade_model.parameters(), lr=0.001)

# 训练模型
cascade_model.train()
num_epochs = 10

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cascade_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试模型
cascade_model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = cascade_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
