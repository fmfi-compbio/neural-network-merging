import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from teacher import Teacher
from dataset import DataLoader
from plot_history import plot_history

from tqdm import tqdm


def train_teacher(teacher_num):
    device = torch.device('cpu')
    train_loader = DataLoader(32, device)
    test_loader = DataLoader(32, device)

    model = Teacher()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[800], gamma=0.1)

    history = []
    for _ in tqdm(range(900)):
        model.train()
        running_training_loss = 0

        for data, targets in train_loader():
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()

        model.eval()
        running_test_loss = 0

        for data, targets in test_loader():
            outputs = model(data)
            loss = F.mse_loss(outputs, targets)

            running_test_loss += loss.item()

        scheduler.step()
        history.append((running_training_loss / len(train_loader), running_test_loss / len(test_loader)))

    torch.save(model.state_dict(), 'teachers/long_teacher{}.pt'.format(teacher_num))
    plot_history(history, [0, 1], ['train loss', 'test loss'], 'teachers/long_teacher{}_losses.png'.format(teacher_num))


def main():
    for teacher_num in range(50):
        train_teacher(teacher_num)


if __name__ == '__main__':
    main()
