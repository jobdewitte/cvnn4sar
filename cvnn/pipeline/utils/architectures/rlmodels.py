from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50
from torchvision.models.detection.fcos import FCOS
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def real_resnet(
        name: str,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs: Any
        ) -> ResNet:
    	
        if in_channels == None:
             in_channels == 4
        if num_classes == None:
             num_classes = 3

        if name == 'resnet18':
             model = resnet18(num_classes=num_classes)
             model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        elif name == 'resnet34':
             model = resnet34(num_classes=num_classes)
             model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif name == 'resnet50':
             model = resnet50(num_classes=num_classes)
             model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
             
        else:
            raise ValueError('Invalid ResNet version. Choose from [`resnet18`, `resnet34`, `resnet50`].')
        
        return model

def real_fcos_resnet_fpn(
    backbone_name: str,
    in_channels: Optional[int] = 4,
    num_classes: Optional[int] = 4,
    frozen_backbone: Optional[str] = None,
    
    **kwargs: Any
    ) -> FCOS:


    backbone = real_resnet(backbone_name, in_channels = in_channels, num_classes= num_classes-1)
    trainable_backbone_layers = 5
    
    if not frozen_backbone is None:
         backbone.load_state_dict(torch.load(frozen_backbone))
         trainable_backbone_layers = 0
    
    backbone_with_fpn = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], 
        extra_blocks=LastLevelP6P7(256, 256), norm_layer=nn.BatchNorm2d
        )
    
    model = FCOS(backbone_with_fpn, num_classes, **kwargs)

    return model