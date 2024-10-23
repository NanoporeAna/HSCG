import torch
from torch import nn
from types import MethodType, FunctionType

from torch.nn import init


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class ResBlock(nn.Module):
    def __init__(self,input_channels, out_channels):
        super(ResBlock, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.input_channels, self.out_channels, kernel_size=1, stride=2, padding=0)
        self.conv2 = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv2(x)
        y = self.relu(y)
        y = self.conv3(y)
        x = self.conv1(x)

        return x + y


class SqueezeNet(nn.Module):
    def __init__(self, version=1.1, num_classes=768):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            # 原始
            # 改成1通道 nn.Conv2d(1, 64, kernel_size=3, stride=2)
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
            # Final convolution is initialized differently form the rest
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)

        elif version == 1.1:
            # 删除
            # 改成1通道 nn.Conv2d(1, 8, kernel_size=3, stride=2)
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
            )
            # Final convolution is initialized differently form the rest
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        elif version == 1.2:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),  # 额外添加一个卷积层，网络通道数开始变化不那么大
                nn.Conv2d(16, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        elif version == 1.3:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  # 删除策略
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 额外添加pool降低输出大小
                Fire(512, 64, 256, 256),
                Fire(512, 128, 512, 512)
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        elif version == 1.4:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),  # 额外添加一个卷积层，网络通道数开始变化不那么大
                nn.Conv2d(16, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  # 删除策略
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 额外添加pool降低输出大小
                Fire(512, 64, 256, 256),
                Fire(512, 128, 512, 512)
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        # 在1.4的基础上后边增加两层fire
        elif version == 1.5:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),  
                nn.Conv2d(16, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(512, 64, 128, 128),
                Fire(256, 64, 128, 128),
                Fire(256, 128, 256,256),
                Fire(512, 128, 512, 512)
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        # 在1.4的基础上后边增加4层fire
        elif version == 1.6:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),  
                nn.Conv2d(16, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(512, 64, 128, 128),
                Fire(256, 64, 128, 128),
                Fire(256, 128, 256,256),
                Fire(512, 128, 512, 512),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(1024, 256, 512, 512),
                Fire(1024, 256, 512, 512),
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        # 对1.4的前边两层卷乘集进行更改，添加残差处理
        elif version == 1.7:
            self.features = nn.Sequential(
                ResBlock(input_channels=1, out_channels=64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  # 删除策略
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 额外添加pool降低输出大小
                Fire(512, 64, 256, 256),
                Fire(512, 128, 512, 512)
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        elif version == 1.8:#在1.4的基础上后边增加4层fire
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),  
                nn.Conv2d(16, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(512, 64, 128, 128),
                Fire(256, 64, 128, 128),
                Fire(256, 128, 256,256),
                Fire(512, 128, 512, 512),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(1024, 128, 256, 256),
                Fire(512, 128, 256, 256),
                Fire(512, 128, 512, 512),
                Fire(1024, 128, 512, 512)
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        elif version == 1.9:#在1.4的基础上后边增加4层fire
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),  
                nn.Conv2d(16, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(512, 64, 128, 128),
                Fire(256, 64, 128, 128),
                Fire(256, 128, 256,256),
                Fire(512, 128, 512, 512),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  
                Fire(1024, 128, 256, 256),
                Fire(512, 128, 256, 256),
                Fire(512, 128, 512, 512),
                Fire(1024, 128, 512, 512),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), 
                Fire(1024, 256, 512, 512),
                Fire(1024, 256, 512, 512)
            )
            final_conv = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.dense = nn.Linear(self.num_classes, self.num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), self.num_classes)
        x = self.dense(x)
        return x

