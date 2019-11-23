import torch.nn as nn
from torchvision import models

n_classes = 20

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, n_classes)

    def forward(self, x):
        return self.model(x)

class ResNext(nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, n_classes)

    def forward(self, x):
        return self.model(x)

