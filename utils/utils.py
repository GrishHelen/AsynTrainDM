import random

import numpy as np
import torch
from enum import Enum

class FinetuneTsType(Enum):
    CONST = 1
    CONST_DELTA = 2
    BLOCK_2X2 = 3
    RANDOM = 4


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
