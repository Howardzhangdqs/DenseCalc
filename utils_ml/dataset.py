import torch
from torch.utils.data import Dataset
from torchvision import transforms
from os import path
import numpy as np

from typing import Tuple, List

try:
    import sys
    sys.path.append(path.join(path.dirname(__file__), '..'))
    import utils
    from config import config
except ImportError:
    from .. import utils


_, dataset_path = config.RAW_DATASET_PATH, config.DATASET_PATH

train_dataset_path = path.join(dataset_path, 'train_data')
test_dataset_path = path.join(dataset_path, 'test_data')


def size_to_embedding(size: int) -> int:
    if size == 16:
        return 0
    elif size == 32:
        return 1
    elif size == 64:
        return 2
    elif size == 128:
        return 3
    elif size == 256:
        return 4
    elif size == 512:
        return 5
    elif size == 1024:
        return 6


class DenseCalcDataset(Dataset):
    def __init__(self, train=True, transform=None, N=None):
        self.transform = config.transform() if transform is None else transform
        self.dataset = utils.ImageFS(train_dataset_path if train else test_dataset_path)
        self.N = N

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, List[int]]:
        mat, note = self.dataset.get_image(index)

        note = utils.decode_dict(note)

        embedding = size_to_embedding(note[1])

        if self.N is not None:
            note = [
                1 if note[0] > self.N else 0,
                0 if note[0] > self.N else 1,
            ]
        else:
            note = [
                note[0],
                0,
            ]

        mat = self.transform(mat)

        # embedding 与 mat 相加
        mat = mat + embedding

        return mat, note


if __name__ == '__main__':
    from rich import print

    dataset = DenseCalcDataset(N=2)
    print(len(dataset))

    from torch.utils.data import DataLoader
    import time

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, (img, note) in enumerate(dataloader):
        combined = torch.stack((note[0], note[1], note[2]), dim=1)
        print(img.shape, combined)
        if i > 3:
            break

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    TEST_ROUNDS = 1

    # sys.exit(0)

    start_time = time.time()
    for i, (img, note) in enumerate(dataloader):
        combined = torch.stack((note[0], note[1]), dim=1)
        # print(img.shape, combined)
        if i >= TEST_ROUNDS - 1:
            break
    end_time = time.time()

    print(f"Read speed: {((end_time - start_time) / TEST_ROUNDS):.4f} seconds")
