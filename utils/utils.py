import random
from enum import Enum

import numpy as np
import torch


class FinetuneTsType(Enum):
    CONST = 'constant'
    CONST_DELTA = 'constant_delta'
    BLOCK_2X2 = 'block_2x2'
    RANDOM = 'random'


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
