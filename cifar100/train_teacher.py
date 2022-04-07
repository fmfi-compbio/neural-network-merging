import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from teacher import Teacher
from plot_history import plot_history

from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import sys

def train_teachers(teacher_num):
    torch.manual_seed(teacher_num)
    device = torch.device('cuda:0')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)

    model = Teacher()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    history = []
    for epoch in tqdm(range(200)):
        model.train()
        running_training_loss = 0
        correct_answers_train = 0
        total_train = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_answers_train += (predicted == targets).sum().item()
            total_train += len(targets)

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

        scheduler.step()
        print(epoch, (running_training_loss / len(train_loader), running_test_loss / len(test_loader),
                        correct_answers_train / total_train,
                        correct_answers_test / total_test))

        history.append((running_training_loss / len(train_loader), running_test_loss / len(test_loader),
                        correct_answers_train / total_train,
                        correct_answers_test / total_test))

    torch.save(model.state_dict(), 'teachers/teacher{}.pt'.format(teacher_num))
    plot_history(history, [0, 1], ['train loss', 'test loss'], 'teachers/teacher{}_losses.png'.format(teacher_num))
    plot_history(history, [2, 3], ['train acc', 'test acc'], 'teachers/teacher{}_acc.png'.format(teacher_num))


if __name__ == '__main__':
    train_teachers(int(sys.argv[1]))
