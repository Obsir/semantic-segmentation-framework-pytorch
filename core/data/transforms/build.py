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
