from tqdm import tqdm
import shutil
import glob
import os
import os.path as path
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from matplotlib import pyplot as plt
from EasyObj import BetterList

from rich import print as rprint

from typing import List, Tuple

from config import config
import utils


_raw_dataset_path, target_dataset_path = config.RAW_DATASET_PATH, config.DATASET_PATH

IMG_SIZE = 256
IMG_CROP_SIZES = BetterList([(512, 512), (256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]).map(lambda x: (f"{x[0]}x{x[1]}", x))

rprint(IMG_CROP_SIZES)


# 单个分区内存储的数据量
IMG_IN_STRAGE_MAX = 64

# 调试模式
DEBUG = False

# 采样频率
SAMPLE_FREQ = 1


datasets = os.listdir(_raw_dataset_path)

rprint(datasets)


for file in glob.glob(f'{target_dataset_path}/*'):
    if os.path.isfile(file):
        os.remove(file)
    else:
        shutil.rmtree(file)


def fileid2filename(fileid: int):
    return (
        fileid // IMG_IN_STRAGE_MAX,
        fileid % IMG_IN_STRAGE_MAX,
        fileid % IMG_IN_STRAGE_MAX == IMG_IN_STRAGE_MAX - 1
    )


def read_image_and_mat(folder_path: str, index: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    img = cv2.imread(path.join(folder_path, 'images', f'IMG_{index}.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ground_truth = loadmat(path.join(folder_path, 'ground_truth', f'GT_IMG_{index}.mat'))["image_info"][0][0][0][0][0]
    ground_truth = list(map(lambda x: (int(x[0]), int(x[1])), ground_truth))

    return img, ground_truth


def list_files(folder_path: str) -> List[str]:
    images = map(lambda x: int(x.lower().replace('.jpg', '').replace('img_', '')), os.listdir(path.join(folder_path, 'images')))
    images = sorted(images)
    return images


for dataset_name in datasets:

    dataset_path = path.join(_raw_dataset_path, dataset_name)
    output_path = path.join(target_dataset_path, dataset_name)

    print(f"Processing {dataset_name}, saving to {output_path}")

    file_index = list_files(dataset_path)

    image_fs = utils.ImageFS(output_path, compress=False, images_per_batch=1)

    for size_index, (size_name, size) in enumerate(IMG_CROP_SIZES):
        retention_probability = 0.5 * (0.7 ** size_index)
        print(f"Processing {dataset_name} with size {size_name}, retention_probability={retention_probability:.4f}")

        first_batch = True
        saved_images_count = 0

        for index in tqdm(file_index, desc="Processing files"):

            img, gt = read_image_and_mat(dataset_path, index)

            crops = utils.split_image_to_patches(
                img, size, (size[0] // 3, size[1] // 3), gt,
                max_patch_num=2 if "test" in dataset_name else 10,
                retention_probability=retention_probability
            )

            if first_batch and DEBUG:
                plt.figure(figsize=(20, 20))

            for index, crop in enumerate(crops):

                targets_in_crop = crop[2]
                cropped_img = cv2.resize(crop[0], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

                image_fs.save_image(cropped_img, utils.encode_dict([targets_in_crop, size[0]]))
                saved_images_count += 1

                if first_batch and index < 10 and DEBUG:
                    plt.subplot(1, 10, index + 1)
                    plt.imshow(cropped_img)
                    plt.title(f"{targets_in_crop} targets")
                    plt.axis('off')

            if first_batch and DEBUG:
                plt.show()
                first_batch = False

            if DEBUG:
                break

        print(f"Saved {saved_images_count} images for size {size_name}")
        image_fs.save_index()
    image_fs.save_index()

    if DEBUG:
        break
