import torch
from torch import Tensor
import torch.nn as nn


class Teacher(nn.Module):
    def __init__(self) -> None:
        super(Teacher, self).__init__()
        self.lin1 = nn.Linear(1, 100)
        self.lin2 = nn.Linear(100, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x
