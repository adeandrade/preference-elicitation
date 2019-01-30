import math
import os
import random
from typing import Tuple, Generator

import numba
import numpy as np


@numba.jitclass([
    ('user_indptr', numba.int32[:]),
    ('user_indices', numba.int32[:]),
    ('num_users', numba.int32),
    ('num_items', numba.int32),
    ('min_positive_items_per_user', numba.int32),
    ('num_items_per_user', numba.int32),
    ('split_index', numba.int32)])
class UserInteractionDataset(object):
    def __init__(
            self,
            user_indptr: np.ndarray,
            user_indices: np.ndarray,
            train_ratio: float = 0.9):
        """

        :param user_indptr:
        :param user_indices:
        :param train_ratio:
        """
        self.user_indptr = user_indptr
        self.user_indices = user_indices

        self.num_users = self.user_indptr.size - 1
        self.num_items = np.max(self.user_indices) + 1

        self.min_positive_items_per_user = self.calculate_items_per_user()
        self.num_items_per_user = self.min_positive_items_per_user * 2

        self.split_index = int(math.ceil(self.num_items * train_ratio))

    def calculate_items_per_user(self) -> int:
        """

        :return:
        """
        minimum = 2 ^ 31 - 1

        for index in range(self.user_indptr.size - 1):
            minimum = min(minimum, self.user_indptr[index + 1] - self.user_indptr[index])

        return minimum

    def get_train_batches(self, size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Creates a generator of batches of training samples of at most `size`.
        :param size: The maximum size for each batch.
        :return: A generator of tuples with `MultimodalDatasetInputs` inputs and `MultimodalDatasetTargets` targets.
        """
        for start_index in range(0, self.split_index, size):
            end_index = min(start_index + size, self.split_index)

            yield self.get_interaction_subset(start_index, end_index)

    def get_validation_batches(self, size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Creates a generator of batches of validation samples of at most `size`.
        :param size: The maximum size for each batch.
        :return: A generator of tuples with `MultimodalDatasetInputs` inputs and `MultimodalDatasetTargets` targets.
        """
        for start_index in range(self.split_index, self.num_users, size):
            end_index = min(start_index + size, self.num_users)

            yield self.get_interaction_subset(start_index, end_index)

    def get_interaction_subset(self, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param start_index:
        :param end_index:
        :return:
        """
        accumulated_lengths = self.user_indptr[start_index:end_index + 1]

        positive_indices = self.user_indices[self.user_indptr[start_index]:self.user_indptr[end_index]]

        negative_indices = self.get_negative_user_indices(start_index, end_index)

        return self.interleave_indices(accumulated_lengths, positive_indices, negative_indices)

    def get_negative_user_indices(self, start_index: int, end_index: int) -> np.ndarray:
        """

        :param start_index:
        :param end_index:
        :return:
        """
        num_examples = self.user_indptr[end_index] - self.user_indptr[start_index]

        negative_user_indices = np.empty(num_examples, np.int32)

        example_index = 0

        for item_index in range(start_index, end_index):
            positive_user_indices = self.user_indices[self.user_indptr[item_index]:self.user_indptr[item_index + 1]]

            num_user_indices = positive_user_indices.size

            for _ in range(num_user_indices):
                negative_user_index = random.randrange(0, self.num_users)
                index = 0
                forward = True

                while index < num_user_indices:
                    positive_user_index = positive_user_indices[index]

                    if positive_user_index < negative_user_index:
                        index += 1
                    elif positive_user_index > negative_user_index:
                        break
                    elif forward and positive_user_index < self.num_users - 1:
                        negative_user_index += 1
                        index += 1
                    else:
                        forward = False
                        negative_user_index -= 1
                        index -= 1

                negative_user_indices[example_index] = negative_user_index
                example_index += 1

        return negative_user_indices

    def interleave_indices(
            self,
            accumulated_lengths: np.ndarray,
            positive_indices: np.ndarray,
            negative_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param accumulated_lengths:
        :param positive_indices:
        :param negative_indices:
        :return:
        """
        batch_size = accumulated_lengths.size - 1

        matrix_shape = (batch_size, self.min_positive_items_per_user * 2)

        user_indices = np.empty(matrix_shape, dtype=np.int32)
        targets = np.empty(matrix_shape, dtype=np.float32)

        for batch_index in enumerate(batch_size):
            offset = accumulated_lengths[batch_index]

            length = accumulated_lengths[batch_index + 1] - offset

            positive_sparse_indices = np.arange(length) + offset
            np.random.shuffle(positive_sparse_indices)

            for pair_index in range(self.min_positive_items_per_user):
                item_index = pair_index * 2

                user_indices[batch_index, item_index] = positive_indices[positive_sparse_indices[item_index]]
                user_indices[batch_index, item_index + 1] = negative_indices[offset + item_index]

                targets[batch_index, item_index] = 1.0
                targets[batch_index, item_index + 1] = 0.0

        return user_indices, targets


def load_item_embeddings(path: str, num_items: int) -> np.ndarray:
    """
    Loads item embeddings as a read-only memory map.
    The number of items in the embedding must match the file size.
    :param path: The path to the data.
    :param num_items: The number of items in the embedding representations.
    :return: A matrix of size (embedding size, number of dimensions).
    """
    embeddings_path = os.path.join(path, 'item_embeddings.bin')

    num_dimensions = os.path.getsize(embeddings_path) // (num_items * 4)

    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode='r',
        shape=(num_items, num_dimensions))

    return embeddings


def load_dataset(path: str, train_ratio: float = 0.9):
    """

    :param path:
    :param train_ratio:
    :return:
    """
    user_indptr = np.memmap(os.path.join(path, 'user_indptr.bin'), dtype=np.int32, mode='r')

    user_indices = np.memmap(os.path.join(path, 'user_indices.bin'), dtype=np.int32, mode='r')

    dataset = UserInteractionDataset(user_indptr, user_indices, train_ratio)

    return dataset
