import pickle
import time

import torch


def is_pytorch_1_1_0_or_later():
    return [int(_) for _ in torch.__version__.split(".")[:3]] >= [1, 1, 0]
