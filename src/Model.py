#LIBRARIES
import torch
import segmentation_models_pytorch as SMP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

linkNetModel = SMP.Linknet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=32,
)
