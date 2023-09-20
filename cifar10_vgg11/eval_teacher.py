import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from plot_history import plot_history
#from vgg import VGG
from teacher import Teacher

from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import sys
    
device = torch.device('cuda:0')
normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        std = (0.2023, 0.1994, 0.2010))
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True)

#model = VGG('VGG11', 10, batch_norm=False, bias=False, relu_inplace=True) 
model = Teacher()
print(model)
model.to(device)

model.load_state_dict(torch.load(sys.argv[1], map_location=device)["model_state_dict"])

model.eval()
running_test_loss = 0
correct_answers_test = 0
total_test = 0

for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)
    outputs = model(data)
    loss = F.cross_entropy(outputs, targets)

    running_test_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    correct_answers_test += (predicted == targets).sum().item()
    total_test += len(targets)

print(correct_answers_test / total_test)
