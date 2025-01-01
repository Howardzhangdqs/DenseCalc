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

        if self.N is not None:
            note = [
                1 if note[0] > self.N else 0,
                0 if note[0] > self.N else 1,
                note[1],
            ]
        else:
            note = [
                note[0],
                0,
                note[1],
            ]

        return self.transform(mat), note


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
