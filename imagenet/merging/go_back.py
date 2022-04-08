import torch
from train_student import initialize_student
import sys

sd = torch.load('../output/resnet18_a3_0-40c531c8.pth')

stud = initialize_student()
stud.compress()
stud.load_state_dict(torch.load(sys.argv[1]))

k_ours = [k for k in stud.state_dict()]
k_theirs = [k for k in sd]

sd2 = {}

for a, b in zip(k_ours, k_theirs):
    if stud.state_dict()[a].shape != sd[b].shape:
        print(a, b, stud.state_dict()[a].shape, sd[b].shape)
    sd2[b] = stud.state_dict()[a]

torch.save(sd2, sys.argv[2])


