import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt

data_dir = '/data/martin/imagewoof2-160'


class TrainLoader:
    def __init__(self, batch_size, num_workers, device):
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_transforms = tt.Compose([
            tt.RandomResizedCrop((224, 224), scale=(0.4, 0.8)),
            tt.RandomHorizontalFlip(),
            tt.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)])

        train_dataset = ImageFolder(data_dir + '/train', train_transforms)
        self.length = len(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       pin_memory=True, prefetch_factor=batch_size // num_workers)
        self.device = device

    def __call__(self, *args, **kwargs):
        for data, targets in self.train_loader:
            yield data.to(self.device), targets.to(self.device)

    def __len__(self):
        return len(self.train_loader)

    def n_samples(self):
        return self.length


class TestLoader:
    def __init__(self, batch_size, num_workers, device):
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        test_transforms = tt.Compose([
            tt.Resize((224, 224)),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)])

        test_dataset = ImageFolder(data_dir + '/val', test_transforms)
        self.length = len(test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, pin_memory=True)
        self.device = device

    def __call__(self, *args, **kwargs):
        for data, targets in self.test_loader:
            yield data.to(self.device), targets.to(self.device)

    def __len__(self):
        return len(self.test_loader)

    def n_samples(self):
        return self.length


def download_dataset():
    # download the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
    download_url(dataset_url, '..')

    # extract the archive

    with tarfile.open('../imagewoof2-160.tgz', 'r:gz') as tar:  # read file in r mode
        tar.extractall(path='..')  # extract all folders from zip file and store under folder named data
