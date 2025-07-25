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
    def __init__(self, directory, batch_size, context_length, mode="train", rank=0, world_size=1, eval_iterations=None):
        self.npy_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
        self.batch_size = batch_size
        self.context_length = context_length
        self.file_lengths = []
        self.arrays = []
        for f in self.npy_files:
            arr = np.load(f, mmap_mode="r")
            self.file_lengths.append(len(arr))
            self.arrays.append(arr)
        self.offsets = [0 for _ in self.npy_files]
        self.max_offsets = [l - context_length - 1 for l in self.file_lengths]
        self.finished = [False for _ in self.npy_files]
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        self.eval_iterations = eval_iterations
        # Only build index in eval mode
        if self.mode == "eval":
            self.eval_indices = []
            for file_idx, arr in enumerate(self.arrays):
                max_offset = self.max_offsets[file_idx]
                for offset in range(0, max_offset + 1, self.context_length):
                    self.eval_indices.append((file_idx, offset))
            # Split indices among different ranks
            self.eval_indices = self.eval_indices[self.rank::self.world_size]
            self.eval_ptr = 0
            self.eval_batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "eval":
            # In eval mode: sequentially iterate over all samples assigned to this rank
            if self.eval_iterations is not None and self.eval_batch_count >= self.eval_iterations:
                raise StopIteration
            if self.eval_ptr >= len(self.eval_indices):
                raise StopIteration
            batch_x, batch_y = [], []
            for _ in range(self.batch_size):
                if self.eval_ptr >= len(self.eval_indices):
                    break
                file_idx, offset = self.eval_indices[self.eval_ptr]
                arr = self.arrays[file_idx]
                x = arr[offset:offset+self.context_length]
                y = arr[offset+1:offset+self.context_length+1]
                batch_x.append(x)
                batch_y.append(y)
                self.eval_ptr += 1
            if not batch_x:
                raise StopIteration
            self.eval_batch_count += 1
            return np.stack(batch_x), np.stack(batch_y)
        else:
            # In train mode: original logic, random sampling
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

