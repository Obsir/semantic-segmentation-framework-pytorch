import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.ENCODER = "vgg16"  #
_C.MODEL.ARCHITECTURE = 'Unet'  # [Unet, Linknet, FPN, PSPNet, PAN, DeepLabV3]
_C.MODEL.ACTIVATION = 'softmax2d'
_C.MODEL.ENCODER_WEIGHTS = 'imagenet'
_C.MODEL.WEIGHT = ''
_C.MODEL.DEVICE = "cuda"
_C.MODEL.GPU_NUM = 0
_C.MODEL.LOSS = "DiceLoss"
_C.MODEL.METRICS = ("Dice", )
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE = 512

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.VAL = ()
_C.DATASETS.TEST = ()
_C.DATASETS.NUM_CLASS = -1
_C.DATASETS.IGNORE_CHANNELS = ()
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SIZE_DIVISIBILITY = 32
#
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 80
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
# the learning rate factor of deformable convolution offsets
_C.SOLVER.DCONV_OFFSETS_LR_FACTOR = 1.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.IMS_PER_BATCH_TRAIN = 8
_C.SOLVER.IMS_PER_BATCH_VAL = 1
_C.SOLVER.IMS_PER_BATCH_TEST = 1
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
