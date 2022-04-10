import os
from dataset import download_dataset

if __name__ == '__main__':
    download_dataset()
    os.mkdir('./students')
    os.mkdir('./teachers')
