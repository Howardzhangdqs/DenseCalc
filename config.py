import os.path as path
import torch
from torchvision import transforms

import utils


class DefaultConfig:

    def __init__(self):
        self.RAW_DATASET_PATH: str = path.normpath(path.join("/home/jinhao/shanghaitech", "part_B_final"))
        self.DATASET_PATH: str = "/ssd2/jinhao/densecalc/dataset"

    def transform(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def inverse_transform(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        return transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=1/std),
            transforms.Normalize(mean=-mean, std=[1, 1, 1])
        ])


config = DefaultConfig()


if __name__ == '__main__':
    from rich import print
    import numpy as np

    transform = config.transform()
    inverse_transform = config.inverse_transform()

    input = np.random.randint(0, 255, (6, 6, 3)).astype(np.float32)

    print(input)
    input = transform(input)
    print(input)
    input = inverse_transform(input)
    print(input)
