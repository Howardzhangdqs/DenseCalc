import cv2
import numpy as np
from typing import List, Tuple
from ._path import to_absolute_path
import random


all = [
    "split_image_to_patches"
]


def count_target_in_box(box: Tuple[int, int, int, int], gt: List[Tuple[int, int]]) -> int:
    """
    Count the number of targets in the given box.
    """
    x1, y1, x2, y2 = box
    count = 0
    for x, y in gt:
        if x1 <= x <= x2 and y1 <= y <= y2:
            count += 1
    return count


def split_image_to_patches(
    image: np.ndarray,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int] = None,
    gt: List[Tuple[int, int]] = None,
    max_patch_num: int = 200,
    retention_probability: float = 0.1,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
    """
    Splits an image into patches of size patch_size with a given stride. If a patch exceeds the image boundary, it will be padded with zeros.

    Args:
        image (np.ndarray): Input image in cv2 format.
        patch_size (Tuple[int, int]): Size of each patch (height, width).
        stride (Tuple[int, int], optional): Step size for sliding window. Defaults to patch_size.
        max_patch_num (int, optional): Maximum number of patches to generate. Defaults to 100.

    Returns:
        List[Tuple[np.ndarray, Tuple[int, int, int, int]]]: A list of patches and their corresponding coordinates.
        Each patch is represented as a tuple (patch_image, (x1, y1, x2, y2)).
    """
    if stride is None:
        stride = patch_size

    patch_height, patch_width = patch_size
    stride_y, stride_x = stride

    image_height, image_width = image.shape[:2]

    coordinates = []

    for y in range(0, image_height, stride_y):
        for x in range(0, image_width, stride_x):
            x1, y1 = x + random.randint(0, stride_x), y + random.randint(0, stride_y)
            # x1, y1 = x, y
            x2, y2 = x1 + patch_width, y1 + patch_height

            if x1 > image_width or y1 > image_height:
                continue

            coordinates.append((x1, y1, x2, y2))

    # Filter coordinates based on the number of targets in the box
    filtered_coordinates = []
    for coord in coordinates:
        if gt is not None:
            target_count = count_target_in_box(coord, gt)
            if target_count > 0 or (target_count == 0 and random.random() < retention_probability):
                filtered_coordinates.append((coord, target_count))
        else:
            filtered_coordinates.append((coord, 0))

    coordinates = filtered_coordinates

    # Randomly select max_patch_num coordinates
    selected_coordinates = [coordinates[i] for i in np.random.choice(len(coordinates), min(max_patch_num, len(coordinates)), replace=False)]
    # selected_coordinates = sorted(selected_coordinates, key=lambda x: (x[1], x[0]))

    patches = []

    for (x1, y1, x2, y2), target_count in selected_coordinates:
        # Extract the patch with padding if it exceeds the image boundaries
        patch = np.zeros((patch_height, patch_width, image.shape[2]), dtype=image.dtype) \
            if len(image.shape) == 3 else np.zeros((patch_height, patch_width), dtype=image.dtype)
        patch[:max(0, min(patch_height, image_height - y1)), :max(0, min(patch_width, image_width - x1))] = image[y1:y2, x1:x2]

        # Append the patch and its coordinates
        patches.append((patch, (x1, y1, x2, y2), target_count))

    return patches


if __name__ == "__main__":
    # Load an image
    image_path = to_absolute_path("../example/image.png", __file__)
    image = cv2.imread(image_path)

    # Split the image into patches
    patch_size = (512, 512)
    stride = (64, 64)
    patches = split_image_to_patches(image, patch_size, stride)

    import matplotlib.pyplot as plt

    # Display the patches
    num_patches = len(patches)
    num_cols = 10
    num_rows = num_patches // num_cols + 1

    print(f"Number of patches: {num_patches}")

    plt.figure(figsize=(20, 20))
    for i, (patch, (x1, y1, x2, y2)) in enumerate(patches):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        # plt.title(f"({x1}, {y1}, {x2}, {y2})")
        plt.axis("on")

    plt.savefig("patches_with_coordinates.png")
