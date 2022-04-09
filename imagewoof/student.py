from l0module import L0Layer, L0GateLayer2d, concat_first_conv, concat_batch_norm, concat_last_linear, \
    compress_first_conv, compress_batch_norm, compress_final_linear
import torch.nn as nn
import torch
from itertools import chain
import types
from teacher import Teacher


class Student(nn.Module):
    def __init__(self, teacher0, teacher1):
        super(Student, self).__init__()
        self.conv1 = concat_first_conv(teacher0.conv1, teacher1.conv1)
        self.bn1 = concat_batch_norm(teacher0.bn1, teacher1.bn1)
        self.relu = nn.ReLU()
        self.gate = L0GateLayer2d(n_channels=64 * 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = L0Layer(teacher0.layer1, teacher1.layer1, self.gate)
        self.layer2 = L0Layer(teacher0.layer2, teacher1.layer2)
        self.layer3 = L0Layer(teacher0.layer3, teacher1.layer3)
        self.layer4 = L0Layer(teacher0.layer4, teacher1.layer4)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = concat_last_linear(teacher0.fc, teacher1.fc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        mask = self.gate.mask(x)
        x = self.gate(x, mask)
        x = self.maxpool(x)

        x = self.layer1(x, mask)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def l0_loss(self):
        l0_loss = self.gate.l0_loss()
        for layer in self.layers:
            l0_loss += layer.l0_loss()
        return l0_loss

    @staticmethod
    def _new_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

    def compress(self):
        important_indices = self.gate.important_indices()
        self.conv1 = compress_first_conv(self.conv1, important_indices)
        self.bn1 = compress_batch_norm(self.bn1, important_indices)
        for layer in self.layers:
            important_indices = layer.compress(important_indices)
        self.fc = compress_final_linear(self.fc, important_indices)

        delattr(self, 'gate')
        self.forward = types.MethodType(self._new_forward, self)

    def gate_parameters(self):
        parameters = [self.gate.parameters()]
        for layer in self.layers:
            parameters.append(layer.gate_parameters())
        return chain.from_iterable(parameters)

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(), self.bn1.parameters(), self.fc.parameters()]
        for layer in self.layers:
            parameters.append(layer.non_gate_parameters())
        return chain.from_iterable(parameters)

    def gate_values(self):
        values = [self.gate.importance_of_features()]
        for layer in self.layers:
            values += layer.gate_values()

        return values


def compressed_student():
    teacher1 = Teacher()
    teacher1.load_state_dict(torch.load('teachers/teacher0.pt'))

    teacher2 = Teacher()
    teacher2.load_state_dict(torch.load('teachers/teacher1.pt'))

    student = Student(teacher1, teacher2)
    student.compress()
    return student
