# An Efficient Semantic Segmentation on Custom Dataset in PyTorch

![](https://img.shields.io/badge/python-3.6%2B-brightgreen)

This repository aims at providing the necessary building blocks for easily building, training and testing segmentation models on custom dataset using PyTorch.

## Acknowledgments

This repository heavily borrows from「[facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)」and depends on「[qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)」which aims at creating segmentation models with different encoders and decoders.

## Highlights

* Custom datasets can be used for training, validation and testing.
* Hyperparameter can be tuned automatically with the help of NNI (Neural Network Intelligence) developed by Microsoft「[microsoft/nni](https://github.com/microsoft/nni)」
* Highly customized framework.

## Table of content

  1. [Preparation](#preparation)
  2. [Examples](#examples)
       1. [Custom Dataset](#dataset)
       2. [Config](#config)
       3. [Hyperparameter Tuning](#hyperparameter_tuning)
  3. [Models](#models)
     1. [Architectures](#architectires)
     2. [Encoders](#encoders)
        4. [Run](#run)
           1. [Training](#training)
           2. [Testing](#testing)

### Preparation <a name="preparation"></a>

```python
pip install -r requirements.txt
```
### Examples <a name="examples"></a>

### 1. Custom Dataset <a name="dataset"></a>

1. Create a python file to build your custom dataset in [`core/data/datasets/`](core/data/datasets/), for example [`core/data/datasets/custom_dataset.py`](core/data/datasets/custom_dataset.py):

   ```python
   from torch.utils.data import Dataset as BaseDataset
   import cv2
   import os
   import numpy as np
   
   
   class CustomDataset(BaseDataset):
       """CustomDataset. Read images, apply augmentation and preprocessing transformations.
   
       Args:
           images_dir (str): path to images folder
           masks_dir (str): path to segmentation masks folder
           class_values (list): values of classes to extract from segmentation mask
           transforms (albumentations.Compose): data transfromation pipeline
               (e.g. flip, scale, etc.)
           preprocessing (albumentations.Compose): data preprocessing
               (e.g. noralization, shape manipulation, etc.)
   
       """
   
       CLASSES = ['background', 'foreground']
   
       def __init__(
               self,
               images_dir,
               masks_dir,
               classes=None,
               transforms=None,
               preprocessing=None,
               split='train',
       ):
           self.ids = os.listdir(images_dir)
           self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
           self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
   
           # convert str names to class values on masks
           self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
   
           self.augmentation = transforms
           self.preprocessing = preprocessing
           self.split = split
   
   
       def __getitem__(self, i):
   
           # read data
           image = cv2.imread(self.images_fps[i])
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           mask = cv2.imread(self.masks_fps[i], 0)
   
           # extract certain classes from mask (e.g. cars)
           masks = [(mask == v) for v in self.class_values]
           mask = np.stack(masks, axis=-1).astype('float')
   
           # apply augmentations
           if self.augmentation:
               sample = self.augmentation(image=image, mask=mask)
               image, mask = sample['image'], sample['mask']
   
           # apply preprocessing
           if self.preprocessing:
               sample = self.preprocessing(image=image, mask=mask)
               image, mask = sample['image'], sample['mask']
           # The following codes are used for saving predictions in testing phase.
           if self.split == 'test':
          		return image, mask, os.path.basename(self.images_fps[i])
           return image, mask
   
       def __len__(self):
           return len(self.ids)
   ```

2. Add CustomDataset class to [`core/data/datasets/__init__.py`](core/data/datasets/__init__.py):

   ```python
   from .concat_dataset import ConcatDataset
   from .custom_dataset import CustomDataset
   
   __all__ = ["CustomDataset", "ConcatDataset"]
   ```

3. Modify `DatasetCatalog.DATASETS` and corresponding if clause in `DatasetCatalog.get()` in [`core/config/paths_catalog.py`](core/config/paths_catalog.py)

   ```python
   import os
   
   class DatasetCatalog(object):
       DATASETS = {
           "custom_dataset_train": {
               "images_dir": "/path/to/custom_dataset/train/img",
               "masks_dir": "/path/to/custom_dataset/train/mask",
               "classes": ['background', 'foreground'],
               "split": "train",
           },
           "custom_dataset_val": {
               "images_dir": "/path/to/custom_dataset/val/img",
               "masks_dir": "/path/to/custom_dataset/val/mask",
               "classes": ['background', 'foreground'],
               "split": "val",
           },
           "custom_dataset_test": {
               "images_dir": "/path/to/custom_dataset/test/img",
               "masks_dir": "/path/to/custom_dataset/test/mask",
               "classes": ['background', 'foreground'],
               "split": "test",
           },
       }
   
       @staticmethod
       def get(name):
           if 'custom_dataset' in name:
               attrs = DatasetCatalog.DATASETS[name]
               return dict(
                   factory="CustomDataset",
                   args=attrs,
               )
           raise RuntimeError("Dataset not available: {}".format(name))
   ```

4. (Optional) Create your customized testing python file in   [`core/data/datasets/prediction/custom_dataset`](core/data/datasets/prediction/custom_dataset)

   [`core/data/datasets/prediction/custom_dataset/__init__.py`](core/data/datasets/prediction/custom_dataset/__init__.py):

   ```python
   import logging
   from .custom_dataset_prediction import do_custom_dataset_prediction
   
   def custom_dataset_prediction(**kwargs):
       return do_custom_dataset_prediction(
           **kwargs,
       )
   ```

   [`core/data/datasets/prediction/custom_dataset/custom_dataset_prediction.py`](core/data/datasets/prediction/custom_dataset/custom_dataset_prediction.py):

   ```python
   from __future__ import division
   import torch
   import os
   from collections import defaultdict
   import numpy as np
   import segmentation_models_pytorch as smp
   from PIL import Image
   from tqdm import tqdm
   
   
   def do_custom_dataset_prediction(model, data_loader, device, output_folder, logger, dataset_name, **kwargs):
       # You can use different metrics here
       metrics = [
           smp.utils.metrics.Dice(threshold=0.5, ignore_channels=(0,)),  # Ignore the background channel or not
       ]
       test_epoch = smp.utils.train.TestEpoch(
           model=model,
           metrics=metrics,
           device=device,
       )
       for item in test_epoch.run(data_loader):
           if 'predictions' in item.keys() and 'filenames' in item.keys() and 'ground_truth' in item.keys():
               for prediction, ground_truth, file_name in zip(item['predictions'], item['ground_truth'],
                                                              item['filenames']):
                   prediction_mask = np.argmax(prediction, axis=0)
                   ground_truth_mask = np.argmax(ground_truth, axis=0)
                   out_img = Image.fromarray(prediction_mask.astype('uint8'))
                   # out_img.putpalette(custom_palette)
                   # You can save prediction_mask to your specified path.
           else:
               test_logs = item
               str_logs = ['{} - {:.4}'.format(k, v) for k, v in test_logs.items()]
               meters = '\t'.join(str_logs)
               logger.info(
                   '\t'.join(
                       [
                           "Test:",
                           "{meters}",
                           "max mem: {memory:.0f}",
                       ]
                   ).format(
                       meters=meters,
                       memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                   ))
   ```

   [`core/data/datasets/prediction/__init__.py`](core/data/datasets/prediction/__init__.py):

   ```python
   from core.data import datasets
   from .custom_dataset import custom_dataset_prediction
   
   def predict(**kwargs):
       """evaluate dataset using different methods based on dataset type.
       """
       if isinstance(kwargs['data_loader'].dataset, datasets.CustomDataset):
           return custom_dataset_prediction(**kwargs)
       else:
           dataset_name = dataset.__class__.__name__
           raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
   ```

5. (Optional) Choose a couple of augmentation types for your custom dataset in [`core/data/transforms/build.py`](core/data/transforms/build.py):

   ```python
   import albumentations as albu
   
   
   class Transforms(object):
       @staticmethod
       def build_transforms(cfg, split='train'):
           if split == 'train':
               train_transform = [
                   albu.Resize(cfg.INPUT.SIZE, cfg.INPUT.SIZE),
                   # albu.HorizontalFlip(p=0.5),
                   albu.OneOf(
                       [
                           # albu.RandomRotate90(p=1),
                           albu.Rotate(p=1, limit=(-15, 15)),
                       ]
                       , p=0.5),
                   albu.GaussNoise(p=0.5),
                   albu.OneOf(
                       [
                           # albu.CLAHE(p=1),
                           albu.RandomBrightnessContrast(p=1),
                       ],
                       p=0.9,
                   ),
                   albu.OneOf(
                       [
                           albu.IAASharpen(p=1),
                           albu.Blur(p=1),
                           albu.MedianBlur(p=1),
                       ],
                       p=0.9,
                   ),
               ]
               return albu.Compose(train_transform, p=0.6)
           else:
               test_transform = [
                   albu.Resize(cfg.INPUT.SIZE, cfg.INPUT.SIZE)
               ]
               return albu.Compose(test_transform)
   
       @staticmethod
       def to_tensor(x, **kwargs):
           return x.transpose(2, 0, 1).astype('float32')
   
       @staticmethod
       def get_preprocessing(preprocessing_fn):
           """Construct preprocessing transform
   
           Args:
               preprocessing_fn (callbale): data normalization function
                   (can be specific for each pretrained neural network)
           Return:
               transform: albumentations.Compose
   
           """
   
           _transform = [
               albu.Lambda(image=preprocessing_fn),
               albu.Lambda(image=Transforms.to_tensor, mask=Transforms.to_tensor),
           ]
           return albu.Compose(_transform)
   ```

### 2. Config <a name="config"></a>

​	Create a config file in [`configs/`](core/configs/), for example [`configs/Encoder_UNet.yaml`](configs/Encoder_UNet.yaml):

```yaml
MODEL:
  ENCODER: "resnet50"	# Encoder
  ARCHITECTURE: "Unet"	# Decoder
  ACTIVATION: "softmax2d"
  ENCODER_WEIGHTS: "imagenet"
  GPU_NUM: 0
  LOSS: "DiceLoss"
  METRICS: ("Dice", )
DATASETS:
  TRAIN: ("custom_dataset_train",)
  VAL: ("custom_dataset_val",)
  TEST: ("custom_dataset_test",)
  NUM_CLASS: 2
  IGNORE_CHANNELS: (0, )
INPUT:
  SIZE: 512
DATALOADER:
  NUM_WORKERS: 8
  SIZE_DIVISIBILITY: 32
SOLVER:
  BASE_LR: 0.001
  WEIGHT_DECAY: 0.0001
  MAX_EPOCH: 80
  CHECKPOINT_PERIOD: 1
  IMS_PER_BATCH_TRAIN: 8
  IMS_PER_BATCH_VAL: 1
  IMS_PER_BATCH_TEST: 1
```

### 3. Hyperparameter Tuning <a name="hyperparameter_tuning"></a>

​	You can modify [`search_space.json`](./search_space.json) to choose the hyperparameters which will be tuned by NNI in training phase, for example:

```json
{
  "SOLVER.MAX_EPOCH": {
    "_type": "choice",
    "_value": [
      60
    ]
  },
  "SOLVER.IMS_PER_BATCH_TRAIN": {
    "_type": "choice",
    "_value": [
      32
    ]
  },
  "SOLVER.BASE_LR": {
    "_type": "choice",
    "_value": [
      0.0001
    ]
  },
  "MODEL.ENCODER": {
    "_type": "choice",
    "_value": [
      "resnet50",
      "mobilenet_v2",
    ]
  },
  "MODEL.ARCHITECTURE": {
    "_type": "choice",
    "_value": [
      "Unet",
      "FPN",
      "DeepLabV3"
    ]
  }
}
```

### Models <a name="models"></a>

This instruction borrows from「[qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)」, you can find more information there.

#### Architectures <a name="architectires"></a>
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [PSPNet](https://arxiv.org/abs/1612.01105)
 - [PAN](https://arxiv.org/abs/1805.10180)
 - [DeepLabV3](https://arxiv.org/abs/1706.05587)

#### Encoders <a name="encoders"></a>

| Encoder              |               Weights                | Params, M |
| -------------------- | :----------------------------------: | :-------: |
| resnet18             |               imagenet               |    11M    |
| resnet34             |               imagenet               |    21M    |
| resnet50             |               imagenet               |    23M    |
| resnet101            |               imagenet               |    42M    |
| resnet152            |               imagenet               |    58M    |
| resnext50_32x4d      |               imagenet               |    22M    |
| resnext101_32x8d     |        imagenet<br>instagram         |    86M    |
| resnext101_32x16d    |              instagram               |   191M    |
| resnext101_32x32d    |              instagram               |   466M    |
| resnext101_32x48d    |              instagram               |   826M    |
| dpn68                |               imagenet               |    11M    |
| dpn68b               |             imagenet+5k              |    11M    |
| dpn92                |             imagenet+5k              |    34M    |
| dpn98                |               imagenet               |    58M    |
| dpn107               |             imagenet+5k              |    84M    |
| dpn131               |               imagenet               |    76M    |
| vgg11                |               imagenet               |    9M     |
| vgg11_bn             |               imagenet               |    9M     |
| vgg13                |               imagenet               |    9M     |
| vgg13_bn             |               imagenet               |    9M     |
| vgg16                |               imagenet               |    14M    |
| vgg16_bn             |               imagenet               |    14M    |
| vgg19                |               imagenet               |    20M    |
| vgg19_bn             |               imagenet               |    20M    |
| senet154             |               imagenet               |   113M    |
| se_resnet50          |               imagenet               |    26M    |
| se_resnet101         |               imagenet               |    47M    |
| se_resnet152         |               imagenet               |    64M    |
| se_resnext50_32x4d   |               imagenet               |    25M    |
| se_resnext101_32x4d  |               imagenet               |    46M    |
| densenet121          |               imagenet               |    6M     |
| densenet169          |               imagenet               |    12M    |
| densenet201          |               imagenet               |    18M    |
| densenet161          |               imagenet               |    26M    |
| inceptionresnetv2    |   imagenet<br>imagenet+background    |    54M    |
| inceptionv4          |   imagenet<br>imagenet+background    |    41M    |
| efficientnet-b0      |               imagenet               |    4M     |
| efficientnet-b1      |               imagenet               |    6M     |
| efficientnet-b2      |               imagenet               |    7M     |
| efficientnet-b3      |               imagenet               |    10M    |
| efficientnet-b4      |               imagenet               |    17M    |
| efficientnet-b5      |               imagenet               |    28M    |
| efficientnet-b6      |               imagenet               |    40M    |
| efficientnet-b7      |               imagenet               |    63M    |
| mobilenet_v2         |               imagenet               |    2M     |
| xception             |               imagenet               |    22M    |
| timm-efficientnet-b0 | imagenet<br>advprop<br>noisy-student |    4M     |
| timm-efficientnet-b1 | imagenet<br>advprop<br>noisy-student |    6M     |
| timm-efficientnet-b2 | imagenet<br>advprop<br>noisy-student |    7M     |
| timm-efficientnet-b3 | imagenet<br>advprop<br>noisy-student |    10M    |
| timm-efficientnet-b4 | imagenet<br>advprop<br>noisy-student |    17M    |
| timm-efficientnet-b5 | imagenet<br>advprop<br>noisy-student |    28M    |
| timm-efficientnet-b6 | imagenet<br>advprop<br>noisy-student |    40M    |
| timm-efficientnet-b7 | imagenet<br>advprop<br>noisy-student |    63M    |
| timm-efficientnet-b8 |         imagenet<br>advprop          |    84M    |
| timm-efficientnet-l2 |            noisy-student             |   474M    |

 ### Run<a name="run"></a>

### 1. Training<a name="training"></a>

Under the main folder:

```python
nnictl create --config ./nni_config.yml
```

Once you have run successfully, you can get following like interface. For more information, you can visit「[microsoft/nni](https://github.com/microsoft/nni)」.

![image-20200607194154947](.\README.assets\ui.png)

### 2. Testing<a name="testing"></a>

Under the main folder:

```python
python scripts/test_net.py --config ./configs/Encoder_UNet.yaml MODEL.WEIGHT /PATH/TO/BEST/MODEL
```

## TODO
- [ ] Distributed training

