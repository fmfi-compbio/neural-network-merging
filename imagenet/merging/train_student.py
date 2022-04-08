import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import math

from teacher import Teacher
from student import Student
from plot_history import plot_history, plot_gate_values
from tqdm import tqdm


def initialize_student():
    teacher1 = Teacher()
    sd = torch.load('teachers/t1.pth.tar')["state_dict"]

    k_ours = [k for k in teacher1.state_dict()]
    k_loaded = [k for k in sd]
    sd2 = {}
    for a, b in zip(k_ours, k_loaded):
        sd2[a] = sd[b]

    teacher1.load_state_dict(sd2)

    teacher2 = Teacher()
    sdx = torch.load('teachers/t2.pth.tar')["state_dict"]

    k_ours = [k for k in teacher2.state_dict()]
    k_loaded = [k for k in sd]
    sd2 = {}
    for a, b in zip(k_ours, k_loaded):
        if b not in sdx:
            print("miss", b)
            continue
        sd2[a] = sdx[b]

    teacher2.load_state_dict(sd2)

    return Student(teacher2, teacher1)



