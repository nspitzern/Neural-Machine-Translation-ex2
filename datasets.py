from typing import Tuple

import numpy as np
from torch import LongTensor
from torch.utils.data import Dataset

import config_part1


class SentencesDataSet(Dataset):
    def __init__(self, src_lines, trg_lines, src_word2idx, trg_word2idx, shuffle=False, device=config_part1.DEVICE):
        self.src_word2idx = src_word2idx
        self.trg_word2idx = trg_word2idx

        self.shuffle = shuffle
        self.device = device

        self._convert_to_tensors(src_lines, trg_lines)

    def _convert_to_tensors(self, src_lines, trg_lines):
        self.src_tensors = []
        self.trg_tensors = []

        for src_line, trg_line in zip(src_lines, trg_lines):
            # convert each line to a tensor of indices surrounded by start and end tokens
            tokens_list = [self.src_word2idx.get(src_word, self.src_word2idx[config_part1.UNKNOWN_TOKEN]) for src_word in src_line]
            tokens_tensor = LongTensor(tokens_list)
            tokens_tensor = tokens_tensor.to(self.device)
            self.src_tensors.append(tokens_tensor)

            tokens_list = [self.trg_word2idx.get(trg_word, self.trg_word2idx[config_part1.UNKNOWN_TOKEN]) for trg_word in trg_line]
            tokens_tensor = LongTensor(tokens_list)
            tokens_tensor = tokens_tensor.to(self.device)
            self.trg_tensors.append(tokens_tensor)

    def __getitem__(self, index) -> Tuple[LongTensor, LongTensor]:
        return self.src_tensors[index], self.trg_tensors[index]

    def __len__(self):
        return min(len(self.src_tensors), len(self.trg_tensors))

    def __iter__(self):
        return IterSentencesDataSet(self, self.shuffle)


class IterSentencesDataSet:
    def __init__(self, dataset, shuffle=False):
        self.len = len(dataset)
        self.shuffle = shuffle
        self.indices = np.arange(self.len)

        if shuffle:
            np.random.shuffle(self.indices)

        self.indices_idx = 0

        self.data = dataset

    def __iter__(self):
        return self

    def __next__(self):
        # we reached the end of the dataset
        if self.indices_idx >= self.len:
            # go back to the start
            self.indices_idx = 0

            # check if a shuffle of indices is needed
            if self.shuffle:
                np.random.shuffle(self.indices)

            # stop current iteration
            raise StopIteration

        # Not the end yet
        curr_idx = self.indices[self.indices_idx]
        self.indices_idx += 1

        return self.data[curr_idx]
