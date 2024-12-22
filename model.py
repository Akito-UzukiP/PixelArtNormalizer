import torch
import torch.nn as nn
import torch.nn.functional as F

class PSDN(nn.Module):
    def __init__(self, n_pixel_class=24):
        super(PSDN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2) 
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2) 
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1, stride=2)  
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(128, 128)      # 输入为通道数
        self.fc2 = nn.Linear(128, n_pixel_class)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.gap(x).view(x.size(0), -1)  # GAP后展平
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


if __name__ == '__main__':
    model = PSDN()
    x = torch.randn(1, 1, 512, 512)
    y = model(x)
    print(y.shape)  # 输出形状: (1, n_pixel_class)
    print(model.predict(x).shape)  # 输出形状: (1,)
