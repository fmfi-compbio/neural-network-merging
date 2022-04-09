import math
import torch
from torch import nn, Tensor
import types
from itertools import chain


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


def concat_first_conv(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    conv = nn.Conv2d(conv1.in_channels, conv1.out_channels * 2, conv1.kernel_size, conv1.stride, conv1.padding,
                     conv1.dilation, conv1.groups, conv1.bias is not None, conv1.padding_mode)
    conv.weight.data[:conv1.out_channels] = conv1.weight.data.detach().clone()
    conv.weight.data[conv1.out_channels:] = conv2.weight.data.detach().clone()

    if conv1.bias is not None:
        conv.bias.data[:conv1.out_channels] = conv1.bias.data.detach().clone()
        conv.bias.data[conv1.out_channels:] = conv2.bias.data.detach().clone()
    return conv


def compress_first_conv(conv: nn.Conv2d, out_indices: Tensor) -> nn.Conv2d:
    in_indices = torch.arange(conv.in_channels)
    return compress_middle_conv(conv, in_indices, out_indices)


def concat_middle_conv(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    conv = nn.Conv2d(conv1.in_channels * 2, conv1.out_channels * 2, conv1.kernel_size, conv1.stride, conv1.padding,
                     conv1.dilation, conv1.groups, conv1.bias is not None, conv1.padding_mode)
    conv.weight.data *= 0
    conv.weight.data[:conv1.out_channels, :conv1.in_channels] = conv1.weight.data.detach().clone()
    conv.weight.data[conv1.out_channels:, conv1.in_channels:] = conv2.weight.data.detach().clone()

    if conv1.bias is not None:
        conv.bias.data[:conv1.out_channels] = conv1.bias.data.detach().clone()
        conv.bias.data[conv1.out_channels:] = conv2.bias.data.detach().clone()
    return conv


def compress_middle_conv(conv: nn.Conv2d, in_indices: Tensor, out_indices: Tensor) -> nn.Conv2d:
    compressed = nn.Conv2d(len(in_indices), len(out_indices), conv.kernel_size, conv.stride,
                           conv.padding, conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
    compressed.weight.data = conv.weight.data[out_indices][:, in_indices].detach().clone()
    if conv.bias is not None:
        compressed.bias.data = conv.bias.data[out_indices].detach().clone()
    return compressed


def concat_first_linear(lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
    lin = nn.Linear(lin1.in_features, lin1.out_features * 2, lin1.bias is not None)
    lin.weight.data[:lin1.out_features] = lin1.weight.data.detach().clone()
    lin.weight.data[lin1.out_features:] = lin2.weight.data.detach().clone()

    if lin1.bias is not None:
        lin.bias.data[:lin1.out_features] = lin1.bias.data.detach().clone()
        lin.bias.data[lin1.out_features:] = lin2.bias.data.detach().clone()
    return lin


def compress_first_linear(lin: nn.Linear, out_indices: Tensor) -> nn.Linear:
    in_indices = torch.arange(lin.in_features)
    return compress_middle_linear(lin, in_indices, out_indices)


def concat_middle_linear(lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
    lin = nn.Linear(lin1.in_features * 2, lin1.out_features * 2, lin1.bias is not None)

    lin.weight.data *= 0
    lin.weight.data[:lin1.out_features, :lin1.in_features] = lin1.weight.data.detach().clone()
    lin.weight.data[lin1.out_features:, lin1.in_features:] = lin2.weight.data.detach().clone()

    if lin1.bias is not None:
        lin.bias.data[:lin1.out_features] = lin1.bias.data.detach().clone()
        lin.bias.data[lin1.out_features:] = lin2.bias.data.detach().clone()
    return lin


def compress_middle_linear(lin: nn.Linear, in_indices: Tensor, out_indices: Tensor) -> nn.Linear:
    compressed = nn.Linear(len(in_indices), len(out_indices), lin.bias is not None)
    compressed.weight.data = lin.weight.data[out_indices][:, in_indices].detach().clone()
    if lin.bias is not None:
        compressed.bias.data = lin.bias.data[out_indices].detach().clone()
    return compressed


def concat_last_linear(lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
    lin = nn.Linear(lin1.in_features * 2, lin1.out_features, lin1.bias is not None)

    lin.weight.data[:, :lin1.in_features] = lin1.weight.data.detach().clone() / 2
    lin.weight.data[:, lin1.in_features:] = lin2.weight.data.detach().clone() / 2

    if lin1.bias is not None:
        lin.bias.data = lin1.bias.data.detach().clone() / 2 + lin2.bias.data.detach().clone() / 2
    return lin


def compress_final_linear(lin: nn.Linear, in_indices: Tensor) -> nn.Linear:
    out_indices = torch.arange(lin.out_features)
    return compress_middle_linear(lin, in_indices, out_indices)


def concat_batch_norm(bn1: nn.BatchNorm2d, bn2: nn.BatchNorm2d) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(bn1.num_features * 2, bn1.eps, bn1.momentum, bn1.affine, bn1.track_running_stats)

    bn.weight.data[:bn1.num_features] = bn1.weight.data.detach().clone()
    bn.weight.data[bn1.num_features:] = bn2.weight.data.detach().clone()
    if bn1.affine:
        bn.bias.data[:bn1.num_features] = bn1.bias.data.detach().clone()
        bn.bias.data[bn1.num_features:] = bn2.bias.data.detach().clone()
    return bn


def compress_batch_norm(bn: nn.BatchNorm2d, indices: Tensor) -> nn.BatchNorm2d:
    compressed = nn.BatchNorm2d(len(indices), bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
    compressed.weight.data = bn.weight.data[indices]
    if bn.affine:
        compressed.weight.data = bn.weight.data[indices].detach().clone()
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
        return hard_sigmoid(s)

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


class L0Layer(nn.Module):
    def __init__(self, layer1, layer2, main_gate=None):
        super(L0Layer, self).__init__()
        self.relu = nn.ReLU()

        planes = layer1.conv1.out_channels * 2
        self.main_gate = L0GateLayer2d(planes) if main_gate is None else main_gate
        self.using_outside_gate = main_gate is not None

        # downsample
        if layer1.downsample_conv is not None:
            self.downsample_conv = concat_middle_conv(layer1.downsample_conv, layer2.downsample_conv)
            self.downsample_bn = concat_batch_norm(layer1.downsample_bn, layer2.downsample_bn)
        else:
            self.downsample_conv = None

        # first block
        self.conv1 = concat_middle_conv(layer1.conv1, layer2.conv1)
        self.gate1 = L0GateLayer2d(planes)
        self.bn1 = concat_batch_norm(layer1.bn1, layer2.bn1)
        self.conv2 = concat_middle_conv(layer1.conv2, layer2.conv2)
        self.bn2 = concat_batch_norm(layer1.bn2, layer2.bn2)

        # second block
        self.conv3 = concat_middle_conv(layer1.conv3, layer2.conv3)
        self.gate2 = L0GateLayer2d(planes)
        self.bn3 = concat_batch_norm(layer1.bn3, layer2.bn3)
        self.conv4 = concat_middle_conv(layer1.conv4, layer2.conv4)
        self.bn4 = concat_batch_norm(layer1.bn4, layer2.bn4)

    def forward(self, x, main_mask=None):
        main_mask = self.main_gate.mask(x) if main_mask is None else main_mask

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

    @staticmethod
    def _new_forward(self, x):
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
        return out

    def compress(self, in_important_indices):
        out_important_indices = self.main_gate.important_indices().detach()

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_middle_conv(self.downsample_conv, in_important_indices,
                                                        out_important_indices)
            self.downsample_bn = compress_batch_norm(self.downsample_bn, out_important_indices)

        # first block
        important_indices_in_block = self.gate1.important_indices()
        self.conv1 = compress_middle_conv(self.conv1, in_important_indices, important_indices_in_block)
        self.bn1 = compress_batch_norm(self.bn1, important_indices_in_block)
        self.conv2 = compress_middle_conv(self.conv2, important_indices_in_block, out_important_indices)
        self.bn2 = compress_batch_norm(self.bn2, out_important_indices)

        # second block
        important_indices_in_block = self.gate2.important_indices()
        self.conv3 = compress_middle_conv(self.conv3, out_important_indices, important_indices_in_block)
        self.bn3 = compress_batch_norm(self.bn3, important_indices_in_block)
        self.conv4 = compress_middle_conv(self.conv4, important_indices_in_block, out_important_indices)
        self.bn4 = compress_batch_norm(self.bn4, out_important_indices)

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        self.forward = types.MethodType(self._new_forward, self)
        return out_important_indices

    def gate_parameters(self):
        if self.using_outside_gate:
            return chain(self.gate1.parameters(), self.gate2.parameters())
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
        return [self.main_gate.importance_of_features(), self.gate1.importance_of_features(),
                self.gate2.importance_of_features()]
