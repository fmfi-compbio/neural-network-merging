import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import math

from teacher import Teacher
from student import Student
from dataset import DataLoader
from plot_history import plot_history, plot_gate_values
from tqdm import tqdm


def initialize_student(teacher_num1, teacher_num2):
    teacher1 = Teacher()
    teacher1.load_state_dict(torch.load('teachers/teacher{}.pt'.format(teacher_num1)))

    teacher2 = Teacher()
    teacher2.load_state_dict(torch.load('teachers/teacher{}.pt'.format(teacher_num2)))

    return Student(teacher1, teacher2)


def train_student_first_stage(student):
    device = torch.device('cpu')
    train_loader = DataLoader(32, device)
    test_loader = DataLoader(32, device)
    student.to(device)

    optimizer1 = SGD(student.non_gate_parameters(), lr=0.01, momentum=0.9)
    optimizer2 = SGD(student.gate_parameters(), lr=0.1, momentum=0.9)

    history = []
    gate_values = []
    alpha = 0.01
    for _ in tqdm(range(150)):
        alpha += 0.05 * math.sqrt(alpha)
        student.train()
        running_training_main_loss = 0
        running_training_l0_loss = 0

        for data, targets in train_loader():
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            outputs = student(data)
            main_loss = F.mse_loss(outputs, targets)
            l0_loss = alpha * student.l0_loss()
            loss = main_loss + l0_loss
            loss.backward()

            optimizer1.step()
            optimizer2.step()

            running_training_main_loss += main_loss.item()
            running_training_l0_loss += l0_loss.item()

        student.eval()
        running_test_loss = 0

        for data, targets in test_loader():
            outputs = student(data)
            loss = F.mse_loss(outputs, targets)
            running_test_loss += loss.item()

        history.append((running_training_main_loss / len(train_loader), running_training_l0_loss / len(train_loader),
                        running_test_loss / len(test_loader)))
        gate_values.append(student.gate_values())
    return history, gate_values


def train_student_second_stage(student):
    device = torch.device('cpu')
    train_loader = DataLoader(32, device)
    test_loader = DataLoader(32, device)
    student.to(device)

    optimizer = SGD(student.parameters(), lr=0.01, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    history = []
    for _ in tqdm(range(150)):
        student.train()
        running_training_loss = 0

        for data, targets in train_loader():
            optimizer.zero_grad()
            outputs = student(data)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()

        student.eval()
        running_test_loss = 0

        for data, targets in test_loader():
            outputs = student(data)
            loss = F.mse_loss(outputs, targets)
            running_test_loss += loss.item()

        scheduler.step()
        history.append((running_training_loss / len(train_loader), running_test_loss / len(test_loader)))
    return history


def main():
    for student_num in range(50):
        student = initialize_student(student_num * 2, student_num * 2 + 1)
        history, gate_values = train_student_first_stage(student)
        plot_history(history, [0, 1, 2], ['train', 'l0', 'test'],
                     'students/before_compress_losses{}.png'.format(student_num))
        plot_gate_values(gate_values, student_num)

        student.compress()

        history = train_student_second_stage(student)
        plot_history(history, [0, 1], ['train loss', 'test loss'], 'students/student_losses{}.png'.format(student_num))
        torch.save(student.state_dict(), 'students/student{}.pt'.format(student_num))


if __name__ == '__main__':
    main()
