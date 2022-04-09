import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from teacher import Teacher
from dataset import TrainLoader, TestLoader
from plot_history import plot_history

from tqdm import tqdm


def train_teachers():
    device = torch.device('cuda:1')
    train_loader = TrainLoader(32, 4, device)
    test_loader = TestLoader(32, 4, device)

    for teacher_num in [2]:
        model = Teacher()
        model.to(device)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

        history = []
        for _ in tqdm(range(200)):
            model.train()
            running_training_loss = 0
            correct_answers_train = 0

            for data, targets in train_loader():
                optimizer.zero_grad()
                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                running_training_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_answers_train += (predicted == targets).sum().item()

            model.eval()
            running_test_loss = 0
            correct_answers_test = 0

            for data, targets in test_loader():
                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)

                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_answers_test += (predicted == targets).sum().item()

            scheduler.step()
            history.append((running_training_loss / len(train_loader), running_test_loss / len(test_loader),
                            correct_answers_train / train_loader.n_samples(),
                            correct_answers_test / test_loader.n_samples()))

        torch.save(model.state_dict(), 'teachers/teacher{}.pt'.format(teacher_num))
        plot_history(history, [0, 1], ['train loss', 'test loss'], 'teachers/teacher{}_losses.png'.format(teacher_num))
        plot_history(history, [2, 3], ['train acc', 'test acc'], 'teachers/teacher{}_acc.png'.format(teacher_num))


if __name__ == '__main__':
    train_teachers()
