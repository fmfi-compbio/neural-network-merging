from l0module import L0GateLayer1d, concat_first_linear, concat_last_linear, compress_first_linear, \
    compress_final_linear
import torch.nn as nn
import torch
from itertools import chain
import types


class Student(nn.Module):
    def __init__(self, teacher0, teacher1):
        super(Student, self).__init__()
        self.lin1 = concat_first_linear(teacher0.lin1, teacher1.lin1)
        self.gate = L0GateLayer1d(n_features=200)
        self.lin2 = concat_last_linear(teacher0.lin2, teacher1.lin2)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.gate(x)
        x = self.lin2(x)
        return x

    def l0_loss(self):
        return self.gate.l0_loss()

    @staticmethod
    def _new_forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x

    def compress(self):
        important_indices = self.gate.important_indices()
        self.lin1 = compress_first_linear(self.lin1, important_indices)
        self.lin2 = compress_final_linear(self.lin2, important_indices)

        delattr(self, 'gate')
        self.forward = types.MethodType(self._new_forward, self)

    def gate_parameters(self):
        return self.gate.parameters()

    def non_gate_parameters(self):
        return chain(self.lin1.parameters(), self.lin2.parameters())

    def gate_values(self):
        return [self.gate.importance_of_features()]
