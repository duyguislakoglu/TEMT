import torch
from torch.utils.data import Dataset
from typing import Tuple
from torch import Tensor

class TensorDatasetWithMoreNegatives(Dataset[Tuple[Tensor, ...]]):
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, number_of_negatives) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.number_of_negatives = number_of_negatives

    def __getitem__(self, index):
        pack_n = 1 + self.number_of_negatives
        return tuple(tensor[(pack_n*index):(pack_n*index+pack_n)] for tensor in self.tensors)

    def __len__(self):
        return int(len(self.tensors[0]) / (self.number_of_negatives+1))
