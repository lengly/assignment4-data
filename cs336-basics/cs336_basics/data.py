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
    On each next, randomly select a file and sequentially read context_length from the last read position.
    When all files are exhausted, reset offsets and start over. No shuffling within each file.
    Usage:
        loader = UniformMixDataLoader(directory, batch_size, context_length)
        batch_x, batch_y = next(loader)
    """
    def __init__(self, directory, batch_size, context_length):
        self.npy_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
        self.batch_size = batch_size
        self.context_length = context_length
        self.file_lengths = []
        self.arrays = []
        for f in self.npy_files:
            arr = np.load(f, mmap_mode="r")
            self.file_lengths.append(len(arr))
            self.arrays.append(arr)
        # Track the current offset for each file
        self.offsets = [0 for _ in self.npy_files]
        # The maximum valid offset for each file (last possible starting index)
        self.max_offsets = [l - context_length - 1 for l in self.file_lengths]
        # Track whether each file is exhausted
        self.finished = [False for _ in self.npy_files]
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Randomly select a file that is not exhausted, read a chunk of context_length from the current offset,
        and advance the offset. When all files are exhausted, reset offsets and finished flags.
        No shuffling within each file.
        """
        batch_x, batch_y = [], []
        available_files = [i for i, done in enumerate(self.finished) if not done and self.max_offsets[i] >= 0]
        if len(available_files) == 0:
            # All files are exhausted, reset offsets and finished flags
            self.offsets = [0 for _ in self.npy_files]
            self.finished = [False for _ in self.npy_files]
            available_files = [i for i, done in enumerate(self.finished) if not done and self.max_offsets[i] >= 0]
        for _ in range(self.batch_size):
            if not available_files:
                self.offsets = [0 for _ in self.npy_files]
                self.finished = [False for _ in self.npy_files]
                available_files = [i for i, done in enumerate(self.finished) if not done and self.max_offsets[i] >= 0]
            file_idx = random.choice(available_files)
            offset = self.offsets[file_idx]
            arr = self.arrays[file_idx]
            x = arr[offset:offset+self.context_length]
            y = arr[offset+1:offset+self.context_length+1]
            batch_x.append(x)
            batch_y.append(y)
            self.offsets[file_idx] += self.context_length
            # Check if the file is exhausted
            if self.offsets[file_idx] > self.max_offsets[file_idx]:
                self.finished[file_idx] = True
                available_files = [i for i in available_files if i != file_idx]
        return np.stack(batch_x), np.stack(batch_y)

