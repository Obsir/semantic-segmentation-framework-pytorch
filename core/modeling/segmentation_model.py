import segmentation_models_pytorch as smp

model_func = {
    'Unet': smp.Unet,
    'Linknet': smp.Linknet,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'PAN': smp.PAN,
    'DeepLabV3': smp.DeepLabV3,
}


def build_segmentation_model(cfg):
    ENCODER = cfg.MODEL.ENCODER
    ENCODER_WEIGHTS = cfg.MODEL.ENCODER_WEIGHTS
    ACTIVATION = cfg.MODEL.ACTIVATION
    NUM_CLASS = cfg.DATASETS.NUM_CLASS
    ARCHITECTURE = cfg.MODEL.ARCHITECTURE
    model = model_func[ARCHITECTURE](
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=ACTIVATION,
        classes=NUM_CLASS, )

    return model
