import math
import torch
from torch import nn, Tensor
import types
from itertools import chain


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


def connect_first_conv2d(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """first conv is the conv at the beginning of resnet (before layers) that takes 3 channels"""
    conv = nn.Conv2d(3, 64 * 2, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
    conv.weight.data[:64] = conv1.weight.data.detach().clone()
    conv.weight.data[64:] = conv2.weight.data.detach().clone()
    return conv


def connect_middle_conv(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """middle conv is conv in layers/blocks"""
    in_channels = conv1.in_channels
    out_channels = conv1.out_channels
    conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=conv1.kernel_size, stride=conv1.stride,
                     padding=conv1.padding, bias=False)
    conv.weight.data *= 0
    conv.weight.data[:out_channels, :in_channels] = conv1.weight.data.detach().clone()
    conv.weight.data[out_channels:, in_channels:] = conv2.weight.data.detach().clone()
    return conv


def compress_conv2d(conv: nn.Conv2d, in_importance_indices: Tensor, out_important_indices: Tensor) -> nn.Conv2d:
    compressed = nn.Conv2d(len(in_importance_indices), len(out_important_indices), conv.kernel_size,
                           stride=conv.stride, padding=conv.padding, bias=False)
    compressed.weight.data = conv.weight.data[out_important_indices][:, in_importance_indices].detach().clone()
    return compressed

def compress_bn(bn, planes, indices):
    out = nn.BatchNorm2d(planes)
    out.weight.data = bn.weight.data[indices]
    out.bias.data = bn.bias.data[indices]
    return out


def connect_final_linear(lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
    in_features = lin1.in_features
    lin = nn.Linear(in_features * 2, lin1.out_features)
    lin.weight.data *= 0
    lin.weight.data[:, :in_features] = lin1.weight.data.detach().clone() / 2
    lin.weight.data[:, in_features:] = lin2.weight.data.detach().clone() / 2
    lin.bias.data = lin1.bias.data.detach().clone() / 2 + lin2.bias.data.detach().clone() / 2
    return lin


def compress_final_linear(layer: nn.Linear, in_important_indices: Tensor) -> nn.Linear:
    compressed = nn.Linear(len(in_important_indices), layer.out_features)
    compressed.weight.data = layer.weight.data[:, in_important_indices].detach().clone()
    compressed.bias.data = layer.bias.data.detach().clone()
    return compressed


class L0GateLayer2d(nn.Module):
    def __init__(self, n_channels, loc_mean=1, loc_sd=0.01, beta=2 / 3, gamma=-0.1, zeta=1.1, fix_temp=True):
        super(L0GateLayer2d, self).__init__()
        self.n_channels = n_channels
        self.loc = nn.Parameter(torch.zeros(n_channels).normal_(loc_mean, loc_sd))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros((1, n_channels)))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.mask(x)
        x = x.permute(2, 3, 0, 1)
        x = x * mask
        x = x.permute(2, 3, 0, 1)
        return x

    def mask(self, x):
        batch_size = x.shape[0]
        if batch_size > self.uniform.shape[0]:
            self.uniform = torch.zeros((batch_size, self.n_channels), device=self.uniform.device)
        self.uniform.uniform_()
        u = self.uniform[:batch_size]
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
        s = s * (self.zeta - self.gamma) + self.gamma
        probability_of_nonzero_gate = torch.sigmoid(self.loc - self.gamma_zeta_ratio * self.temp)
        return hard_sigmoid(s) # / (probability_of_nonzero_gate + 1e-5)

    def l0_loss(self):
        mean_probability_of_nonzero_gate = torch.mean(torch.sigmoid(self.loc - self.gamma_zeta_ratio * self.temp))
        loss = torch.square(mean_probability_of_nonzero_gate - 0.5)
        return loss

    def importance_of_features(self):
        s = torch.sigmoid(self.loc * (self.zeta - self.gamma) + self.gamma)
        return hard_sigmoid(s)

    def important_indices(self):
        importance = self.importance_of_features().detach()
        important_indices = torch.argsort(importance, descending=True)[:len(importance) // 2]
        return important_indices


def new_forward(self, x):
    identity = x

    # first block
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample_conv is not None:
        identity = self.downsample_conv(identity)
        identity = self.downsample_bn(identity)

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
    return out


class L0Layer(nn.Module):
    def __init__(self, layer1, layer2):
        super(L0Layer, self).__init__()
        self.relu = nn.ReLU()

        planes = layer1.conv1.out_channels * 2
        self.main_gate = L0GateLayer2d(planes)


        # first block
        self.conv1 = connect_middle_conv(layer1.conv1, layer2.conv1)
        self.gate1 = L0GateLayer2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1.weight.data[:planes//2] = layer1.bn1.weight.data.clone()
        self.bn1.weight.data[planes//2:] = layer2.bn1.weight.data.clone()
        self.bn1.bias.data[:planes//2] = layer1.bn1.bias.data.clone()
        self.bn1.bias.data[planes//2:] = layer2.bn1.bias.data.clone()
        self.conv2 = connect_middle_conv(layer1.conv2, layer2.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2.weight.data[:planes//2] = layer1.bn2.weight.data.clone()
        self.bn2.weight.data[planes//2:] = layer2.bn2.weight.data.clone()
        self.bn2.bias.data[:planes//2] = layer1.bn2.bias.data.clone()
        self.bn2.bias.data[planes//2:] = layer2.bn2.bias.data.clone()

        # downsample
        if layer1.downsample_conv is not None:
            self.downsample_conv = connect_middle_conv(layer1.downsample_conv, layer2.downsample_conv)
            self.downsample_bn = nn.BatchNorm2d(planes)
            self.downsample_bn.weight.data[:planes//2] = layer1.downsample_bn.weight.data.clone()
            self.downsample_bn.weight.data[planes//2:] = layer2.downsample_bn.weight.data.clone()
            self.downsample_bn.bias.data[:planes//2] = layer1.downsample_bn.bias.data.clone()
            self.downsample_bn.bias.data[planes//2:] = layer2.downsample_bn.bias.data.clone()
        else:
            self.downsample_conv = None

        # second block
        self.conv3 = connect_middle_conv(layer1.conv3, layer2.conv3)
        self.gate2 = L0GateLayer2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn3.weight.data[:planes//2] = layer1.bn3.weight.data.clone()
        self.bn3.weight.data[planes//2:] = layer2.bn3.weight.data.clone()
        self.bn3.bias.data[:planes//2] = layer1.bn3.bias.data.clone()
        self.bn3.bias.data[planes//2:] = layer2.bn3.bias.data.clone()
        self.conv4 = connect_middle_conv(layer1.conv4, layer2.conv4)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn4.weight.data[:planes//2] = layer1.bn4.weight.data.clone()
        self.bn4.weight.data[planes//2:] = layer2.bn4.weight.data.clone()
        self.bn4.bias.data[:planes//2] = layer1.bn4.bias.data.clone()
        self.bn4.bias.data[planes//2:] = layer2.bn4.bias.data.clone()

    def forward(self, x, main_mask=None):
        if main_mask is None:
            main_mask = self.main_gate.mask(x)

        # downsample
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # first block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gate1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        x = self.main_gate(x, main_mask)

        # second block
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.gate2(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x += identity
        x = self.relu(x)
        x = self.main_gate(x, main_mask)

        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() + self.gate2.l0_loss()

    def compress(self, in_importance_indices):
        out_importance_indices = self.main_gate.important_indices().detach()
        planes = len(out_importance_indices)

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_conv2d(self.downsample_conv, in_importance_indices, out_importance_indices)
            self.downsample_bn = nn.BatchNorm2d(planes)

        # first block
        important_indices_in_block = self.gate1.important_indices()
        self.conv1 = compress_conv2d(self.conv1, in_importance_indices, important_indices_in_block)
        self.bn1 = compress_bn(self.bn1, planes, important_indices_in_block)
        self.conv2 = compress_conv2d(self.conv2, important_indices_in_block, out_importance_indices)
        self.bn2 = compress_bn(self.bn2, planes, out_importance_indices)

        # second block
        important_indices_in_block = self.gate2.important_indices()
        self.conv3 = compress_conv2d(self.conv3, out_importance_indices, important_indices_in_block)
        self.bn3 = compress_bn(self.bn3, planes, important_indices_in_block)
        self.conv4 = compress_conv2d(self.conv4, important_indices_in_block, out_importance_indices)
        self.bn4 = compress_bn(self.bn4, planes, out_importance_indices)

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        self.forward = types.MethodType(new_forward, self)
        return out_importance_indices

    def gate_parameters(self):
        return chain(self.main_gate.parameters(), self.gate1.parameters(), self.gate2.parameters())

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(),
                      self.bn1.parameters(),
                      self.conv2.parameters(),
                      self.bn2.parameters(),
                      self.conv3.parameters(),
                      self.bn3.parameters(),
                      self.conv4.parameters(),
                      self.bn4.parameters()]
        if self.downsample_conv is not None:
            parameters += [self.downsample_conv.parameters(),
                           self.downsample_bn.parameters()]

        return chain.from_iterable(parameters)

    def gate_values(self):
        """used only for plots"""
        return [self.main_gate.importance_of_features(), self.gate1.importance_of_features(),
                self.gate2.importance_of_features()]
