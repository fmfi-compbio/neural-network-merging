import torch
import torch.nn as nn

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        relu_inplace = True
        self.bias = False
        self.features = self._make_layers(
            [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            relu_inplace=relu_inplace)

        self.classifier = nn.Linear(512, 10, bias=self.bias)
        print("Relu Inplace is ", relu_inplace)
 

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, relu_inplace=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                       nn.ReLU(inplace=relu_inplace)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        print("in _make_layers", list(layers))
        return nn.Sequential(*layers)
 
