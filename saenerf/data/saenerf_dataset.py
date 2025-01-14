
import numpy as np
from typing import List, Literal
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from nerfstudio.utils.rich_utils import CONSOLE


class SaENeRFDataset(Dataset):
    """
    input: slice pos/neg frames & pos/neg threshold, 
    output: accumulated frame & split index
    """

    def __init__(self, events_prefix_sum, wind_size_ratio=0.01):
        self.events_prefix_sum = events_prefix_sum
        self.max_win_size = int(len(events_prefix_sum) * wind_size_ratio)
        print("win_size_num", self.max_win_size)
    
    def __len__(self):
        return len(self.events_prefix_sum) - 1
    
    def __getitem__(self, index):
        assert 0 <= index < self.__len__(), f"EventFrame {index} out of range"
        # pre_index = max(0, index - np.random.randint(1, self.wind_size + 1))
        win_size = np.random.randint(1, self.max_win_size + 1)
        # if (win_size & 1) == 0:
        #     win_size //= (win_size & -win_size)
        pre_index = max(0, index - win_size)
        event_frames = self.events_prefix_sum[index + 1] - self.events_prefix_sum[pre_index]
        splits = np.array([pre_index, index], dtype=int)
            
        return {
            "event_frames": event_frames, 
            "splits": splits,
        }