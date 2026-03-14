import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(backbone: str):
    if backbone == "fasterrcnn_resnet50_fpn":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights=weights)

    else:
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, weights=weights)
        

    return model