from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import glob
import os
import random

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
    ])  # fmt: skip
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )  # fmt: skip
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


class UniformMixDataLoader:
    """
    Uniformly mix data from all npy files, ensuring each batch is globally shuffled.
    Each batch consists of samples from all files, with global shuffle of all possible (file, offset) pairs.
    Usage:
        loader = UniformMixDataLoader(directory, batch_size, context_length)
        batch_x, batch_y = next(loader)
    """
    def __init__(self, directory, batch_size, context_length):
        self.npy_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
        self.batch_size = batch_size
        self.context_length = context_length
        self.file_lengths = []
        for f in self.npy_files:
            arr = np.load(f, mmap_mode="r")
            self.file_lengths.append(len(arr))
        self._build_sampling_pool()
        self.arrays = [None for _ in self.npy_files]

    def _build_sampling_pool(self):
        # Build a list of (file_idx, offset) for all files
        self.sampling_pool = []
        for file_idx, length in enumerate(self.file_lengths):
            for offset in range(0, length - self.context_length - 1, self.context_length):
                self.sampling_pool.append((file_idx, offset))
        random.seed(42)
        random.shuffle(self.sampling_pool)
        self.pool_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pool_pos + self.batch_size > len(self.sampling_pool):
            self._build_sampling_pool()  # New epoch
        batch_x, batch_y = [], []
        for _ in range(self.batch_size):
            file_idx, offset = self.sampling_pool[self.pool_pos]
            if self.arrays[file_idx] is None:
                self.arrays[file_idx] = np.load(self.npy_files[file_idx], mmap_mode="r")
            arr = self.arrays[file_idx]
            x = arr[offset:offset+self.context_length]
            y = arr[offset+1:offset+self.context_length+1]
            batch_x.append(x)
            batch_y.append(y)
            self.pool_pos += 1
        return np.stack(batch_x), np.stack(batch_y)

