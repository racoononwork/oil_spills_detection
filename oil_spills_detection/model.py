from typing import Optional
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from config import NUM_CLASSES

import torch
import torchvision



class OilSpillModel:
    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)

        # Update classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Update mask head
        in_channels = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, num_classes)

        self.model.to(self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def get_device(self) -> torch.device:
        return self.device


def get_model_instance(num_classes: int = NUM_CLASSES, weights: bool = True) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights='DEFAULT' if weights else None)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model
