import torch
import math


class DataLoader:
    def __init__(self, batch_size, device, dataset_size=10000):
        self.batch_size = batch_size
        self.device = device
        self.dataset_size = dataset_size
        self.index = 0

        self.X = torch.rand((self.dataset_size, 1))
        self.Y = torch.sin(10 * math.pi * self.X) + torch.normal(mean=0, std=0.2, size=(dataset_size, 1))

    def __call__(self, *args, **kwargs):
        self.index = 0
        while self.index + self.batch_size < self.dataset_size:
            yield self.X[self.index:self.index + self.batch_size].to(self.device), \
                  self.Y[self.index:self.index + self.batch_size].to(self.device)
            self.index += self.batch_size

        if not self.index == self.dataset_size:
            yield self.X[self.index:self.dataset_size].to(self.device), \
                  self.Y[self.index:self.dataset_size].to(self.device)

    def __len__(self):
        return math.ceil(self.dataset_size / self.batch_size)

    def n_samples(self):
        return self.dataset_size
