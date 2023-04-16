import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResidualBlock, self).__init__()
        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.ReLU()(x)
    

class ResNet(nn.Module):
    def __init__(self, in_channels=3, outputs=14):
        super(ResNet, self).__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(48, 48, downsample=False),
            ResidualBlock(48, 48, downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(48, 96, downsample=True),
            ResidualBlock(96, 96, downsample=False)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(96, 192, downsample=True),
            ResidualBlock(192, 192, downsample=False)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(192, 384, downsample=True),
            ResidualBlock(384, 384, downsample=False)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, outputs)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = nn.Flatten()(x)
        x = self.fc(x)

        return x