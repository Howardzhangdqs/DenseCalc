import pandas as pd
import numpy as np
import os
import os.path as path
import sys
from typing import List, Tuple


IMAGES_PER_BATCH = 16


class ImageFS:
    """
    A class to manage a filesystem-based image storage with indexing and caching.
    Attributes:
        image_dir_path (str): Path to the directory where images are stored.
        index_file_path (str): Path to the index file. Defaults to 'index.csv' in the image directory.
        images_per_batch (int): Number of images per batch file. Defaults to IMAGES_PER_BATCH.
        image_shape (Tuple[int, int, int]): Shape of the images. Defaults to None.
        compress (bool): Whether to compress the images. Defaults to True.
        npy_suffix (str): Suffix for the numpy files, either 'npz' or 'npy'.
        dataset (pd.DataFrame): DataFrame containing the index of images.
        npy_index (int): Maximum npy index in the dataset.
        npy_cache (np.ndarray): Cached numpy array of images.
        npy_cache_index (int): Index of the cached numpy file.
        npy_cache_changed (bool): Flag indicating if the cache has changed.
    Methods:
        __len__(): Returns the number of images in the dataset.
        _np_save(file_path: str, data: np.ndarray): Saves the numpy array to a file.
        _np_load(file_path: str) -> np.ndarray: Loads the numpy array from a file.
        _index_to_npyindex_and_infileindex(index: int) -> Tuple[int, int]: Converts the index to npy_index and infile_index.
        _add_new_image(image_note: str) -> int: Adds a new image to the dataset and returns its indices.
        _load_cache(npy_index: int): Loads the cache for the given npy_index.
        save_image(image: np.ndarray, image_note: str): Saves the image to the image directory.
        get_image(index: int) -> Tuple[np.ndarray, str]: Gets the image and its note by index.
        get_image_note(index: int) -> str: Gets the image note by index.
        save_index(): Saves the index file.
    Examples:
        ```python
        # Initialize the ImageFS object
        image_fs = ImageFS('path/to/image_dir', images_per_batch=10)

        # Save an image
        image = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        image_fs.save_image(image, 'Random Image')

        # Get an image by index
        retrieved_image, note = image_fs.get_image(0)
        print(f'Retrieved Image Note: {note}')

        # Save the index file
        image_fs.save_index()
        ```
    """

    def __init__(
        self,
        image_dir_path: str,
        index_file_path: str = None,
        images_per_batch: int = IMAGES_PER_BATCH,
        image_shape: Tuple[int, int, int] = None,
        compress: bool = False
    ):

        self.image_dir_path = image_dir_path
        self.index_file_path = index_file_path if index_file_path else path.join(image_dir_path, 'index.csv')
        self.images_per_batch = images_per_batch
        self.image_shape = image_shape
        self.compress = compress
        self.npy_suffix = 'npz' if compress else 'npy'

        if not os.path.exists(self.image_dir_path):
            os.makedirs(self.image_dir_path, exist_ok=True)

        if not os.path.exists(self.index_file_path):
            os.makedirs(path.dirname(self.index_file_path), exist_ok=True)

        # Load the index file
        self.dataset: pd.DataFrame = pd.read_csv(self.index_file_path) \
            if os.path.exists(self.index_file_path) \
            else pd.DataFrame(columns=['npy_index', 'infile_index', 'note'])

        # Max npy index in the dataset
        self.npy_index = self.dataset['npy_index'].max() if not self.dataset.empty else 0

        # Cache the npy file in memory
        self.npy_cache: np.ndarray = None
        self.npy_cache_index: int = None
        self.npy_cache_changed: bool = False

        if not self.dataset.empty:
            self.image_shape = self.get_image(0)[0].shape

    def __len__(self):
        return len(self.dataset)

    def _np_save(self, file_path: str, data: np.ndarray):
        if self.compress:
            np.savez_compressed(file_path, data=data)
        else:
            np.save(file_path, data)

    def _np_load(self, file_path: str) -> np.ndarray:
        if self.compress:
            return np.load(file_path)['data']
        else:
            return np.load(file_path)

    def _index_to_npyindex_and_infileindex(self, index: int) -> Tuple[int, int]:
        """
        Convert the index to npy_index and infile_index
        """
        return (
            index // self.images_per_batch,
            index % self.images_per_batch,
        )

    def _add_new_image(self, image_note: str) -> int:
        image_index = len(self.dataset)

        npy_index, infile_index = self._index_to_npyindex_and_infileindex(image_index)

        self.dataset = pd.concat([self.dataset, pd.DataFrame([{
            'npy_index': npy_index,
            'infile_index': infile_index,
            'note': image_note
        }])], ignore_index=True)
        return (npy_index, infile_index)

    def _load_cache(self, npy_index: int):
        """
        Load the cache for the given npy_index
        """
        if self.npy_cache_index != npy_index:
            if self.npy_cache is not None and self.npy_cache_changed:
                self._np_save(path.join(self.image_dir_path, f'{self.npy_cache_index}.{self.npy_suffix}'), self.npy_cache)
            npy_file_path = path.join(self.image_dir_path, f'{npy_index}.{self.npy_suffix}')
            self.npy_cache = self._np_load(npy_file_path) \
                if path.exists(npy_file_path) \
                else np.zeros((self.images_per_batch, *self.image_shape), dtype=np.uint8)
            self.npy_cache_index = npy_index
            self.npy_cache_changed = False

    def save_image(self, image: np.ndarray, image_note: str):
        """
        Save the image to the image directory
        """
        if self.image_shape is None:
            self.image_shape = image.shape

        npy_index, infile_index = self._add_new_image(image_note)
        self._load_cache(npy_index)

        self.npy_cache[infile_index] = image
        self.npy_cache_changed = True

    def get_image(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Get the image by index
        """
        npy_index, infile_index = self._index_to_npyindex_and_infileindex(index)
        self._load_cache(npy_index)

        return (self.npy_cache[infile_index], self.dataset.loc[index, 'note'])

    def get_image_note(self, index: int) -> str:
        """
        Get the image note by index
        """
        return self.dataset.loc[index, 'note']

    def save_index(self):
        """
        Save the index file
        """
        if self.npy_cache is not None and self.npy_cache_changed:
            self._np_save(path.join(self.image_dir_path, f'{self.npy_cache_index}.{self.npy_suffix}'), self.npy_cache)
        self.dataset.to_csv(self.index_file_path, index=False)


if __name__ == '__main__':
    image_fs = ImageFS('test', images_per_batch=2)

    import cv2

    # Generate some white images with black numbers
    for i in range(10):
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        cv2.putText(image, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        image = image.transpose(2, 0, 1)
        image_fs.save_image(image, f'Image {i}')

    # Read the images and display them
    for i in range(10):
        image = image_fs.get_image(i)
        image = image.transpose(1, 2, 0)
        cv2.imwrite(f'test/{i}.png', image)
