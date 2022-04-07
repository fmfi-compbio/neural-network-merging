import torch
from torch import Tensor
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Layer(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # downsample
        if stride != 1 or inplanes != planes:
            self.downsample_conv = conv1x1(inplanes, planes, stride)
            self.downsample_bn = nn.BatchNorm2d(planes)
        else:
            self.downsample_conv = None

        # first block
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # second block
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = conv3x3(planes, planes)
        self.bn4 = nn.BatchNorm2d(planes)

        # third block
        self.conv5 = conv3x3(planes, planes)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = conv3x3(planes, planes)
        self.bn6 = nn.BatchNorm2d(planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # first block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        # second block
        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        out += identity
        out = self.relu(out)

        identity = out
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)

        out += identity
        out = self.relu(out)
        return out


class Teacher(nn.Module):
    def __init__(self) -> None:
        super(Teacher, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Layer(16, 16, stride=1)
        self.layer2 = Layer(16, 32, stride=2)
        self.layer3 = Layer(32, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 100)

        # initialization of weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
