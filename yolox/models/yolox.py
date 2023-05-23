#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from PAR.models import MultiTaskHead

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)

#TODO complete class
class YOLOXpar(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, par=None, labels_names=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        if labels_names is None:
            labels_names = [
                "Female",
                "AgeOver60",
                "Age18-60",
                "AgeLess18",
                "Front",
                "Side",
                "Back",
                "Hat",
                "Glasses",
                "HandBag",
                "ShoulderBag",
                "Backpack",
                "HoldObjectsInFront",
                "ShortSleeve",
                "LongSleeve",
                "UpperStride",
                "UpperLogo",
                "UpperPlaid",
                "UpperSplice",
                "LowerStripe",
                "LowerPattern",
                "LongCoat",
                "Trousers",
                "Shorts",
                "Skirt&Dress",
                "Boots"
            ]

        self.backbone = backbone
        self.head = head
        self.par = par
        self.labels_names = labels_names

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)

    def par_img(self, img):
        img = self.backbone.backbone(img)[self.backbone.backbone.out_features[0]]
        gender, pov, sleeve, label = self.par(img)
        gender, pov, sleeve, label = np.squeeze(gender), np.squeeze(pov), np.squeeze(sleeve), np.squeeze(label)
        gender, pov, sleeve, label = torch.sigmoid(gender).cpu().detach().numpy(), torch.sigmoid(pov).cpu().detach().numpy(), torch.sigmoid(sleeve).cpu().detach().numpy(), torch.sigmoid(label).cpu().detach().numpy()
        gender, pov, sleeve, label = np.round(gender), np.round(pov), np.round(sleeve), np.round(label)
        label = np.insert(label, 0, gender)

        label = np.insert(label, 4, pov)

        label = np.insert(label, 13, sleeve)
        return label