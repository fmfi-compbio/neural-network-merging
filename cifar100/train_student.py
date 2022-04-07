import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import math

from teacher import Teacher
from student import Student
from plot_history import plot_history, plot_gate_values
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import sys


def initialize_student(student_num):
    teacher1 = Teacher()
    teacher1.load_state_dict(torch.load('teachers/teacher{}.pt'.format(sys.argv[2])))

    teacher2 = Teacher()
    teacher2.load_state_dict(torch.load('teachers/teacher{}.pt'.format(sys.argv[3])))

    return Student(teacher1, teacher2)


def train_student_first_stage(student):
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
    student.to(device)

    optimizer1 = SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler1 = MultiStepLR(optimizer1, milestones=[50, 70], gamma=0.1)
    optimizer2 = SGD(student.gate_parameters(), lr=0.2, momentum=0.9)
    scheduler2 = MultiStepLR(optimizer2, milestones=[], gamma=1)

    history = []
    gate_values = []
    alpha = 0.05
    for _ in tqdm(range(100)):
        alpha += 0.05 * math.sqrt(alpha)
        student.train()
        running_training_main_loss = 0
        running_training_l0_loss = 0
        correct_answers_train = 0
        total_train = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            outputs = student(data)
            main_loss = F.cross_entropy(outputs, targets)
            l0_loss = alpha * student.l0_loss()
            loss = main_loss + l0_loss
            loss.backward()

            optimizer1.step()
            optimizer2.step()

            running_training_main_loss += main_loss.item()
            running_training_l0_loss += l0_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_answers_train += (predicted == targets).sum().item()
            total_train += len(targets)

        student.eval()
        running_test_loss = 0
        correct_answers_test = 0
        total_test = 0

        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = student(data)
            loss = F.cross_entropy(outputs, targets)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_answers_test += (predicted == targets).sum().item()
            total_test += len(targets)

        scheduler1.step()
        scheduler2.step()

        print((running_training_main_loss / len(train_loader), running_training_l0_loss / len(train_loader),
                        running_test_loss / len(test_loader), correct_answers_train / total_train,
                        correct_answers_test / total_test))

        history.append((running_training_main_loss / len(train_loader), running_training_l0_loss / len(train_loader),
                        running_test_loss / len(test_loader), correct_answers_train / total_train,
                        correct_answers_test / total_test))
        gate_values.append(student.gate_values())

        plot_gate_values(gate_values, int(sys.argv[1]))
    return history, gate_values


def train_student_second_stage(student):
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
    student.to(device)

    optimizer = SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    history = []
    for epoch in tqdm(range(100)):
        student.train()
        running_training_loss = 0
        correct_answers_train = 0
        total_train = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = student(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_answers_train += (predicted == targets).sum().item()
            total_train += len(targets)

        student.eval()
        running_test_loss = 0
        correct_answers_test = 0
        total_test = 0

        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = student(data)
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
    return history


def train_students():
    i = int(sys.argv[1])
    student = initialize_student(i)
    history, gate_values = train_student_first_stage(student)
    plot_history(history, [0, 1, 2], ['train', 'l0', 'test'], 'students/before_compress_losses{}.png'.format(i))
    plot_history(history, [3, 4], ['train acc', 'test acc'], 'students/before_compress_acc{}.png'.format(i))
    plot_gate_values(gate_values, i)

    torch.save(student.state_dict(), 'students/student_bc{}.pt'.format(i))

    student.compress()

    history = train_student_second_stage(student)
    plot_history(history, [0, 1], ['train loss', 'test loss'], 'students/student_losses{}.png'.format(i))
    plot_history(history, [2, 3], ['train acc', 'test acc'], 'students/student_acc{}.png'.format(i))
    torch.save(student.state_dict(), 'students/student{}.pt'.format(i))


if __name__ == '__main__':
    train_students()
