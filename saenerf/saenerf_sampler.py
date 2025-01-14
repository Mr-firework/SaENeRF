import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from nerfstudio.utils.rich_utils import CONSOLE
import timeit
import numba

def saenerf_sampler(event_frames: Tensor, step: int, ray_batch_size: int, neg_ratio: float = 0.05):
    neg_size = int(ray_batch_size * neg_ratio)
    zero_indices_3d = torch.nonzero(event_frames == 0)
    zero_indices_3d_selected = zero_indices_3d[
        np.random.choice(zero_indices_3d.shape[0], size=(neg_size,))
    ]
    nonzero_indices_3d = torch.nonzero(event_frames)
    nonzero_indices_3d_selected = nonzero_indices_3d[
        np.random.choice(nonzero_indices_3d.shape[0], size=(ray_batch_size - neg_size,))
    ]
    indices_3d = torch.concat([zero_indices_3d_selected, nonzero_indices_3d_selected], dim=0)
    #print(zero_indices_2d_selected.shape, nonzero_indices_2d.shape, res.shape)
    return indices_3d
