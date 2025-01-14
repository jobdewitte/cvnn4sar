from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import partial
import torch

from torch import nn
from torchcomplex import nn as cxnn

from torchvisioncomplex.models.resnet import ResNet, resnet18, resnet34, resnet50
from torchvisioncomplex.models.detection.fcos import FCOS
from torchvisioncomplex.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvisioncomplex.ops.feature_pyramid_network import LastLevelP6P7


def complex_resnet(
        name: str,
        in_channels: Optional[int] = 2,
        num_classes: Optional[int] = 3,
        **kwargs: Any
        ) -> ResNet:
    	
        if name == 'resnet18':
             model = resnet18(num_classes=num_classes)
             model.conv1 = cxnn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        elif name == 'resnet34':
             model = resnet34(num_classes=num_classes)
             model.conv1 = cxnn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif name == 'resnet50':
             model = resnet50(num_classes=num_classes)
             model.conv1 = cxnn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
          
        else:
            raise ValueError(f'Invalid ResNet version {name}. Choose from [`resnet18`, `resnet34`, `resnet50`].')
        
        return model

def complex_fcos_resnet_fpn(
    backbone_name: str,
    in_channels: Optional[int] = 2,
    num_classes: Optional[int] = 4,
    frozen_backbone: Optional[str] = None,
    **kwargs: Any
    ) -> FCOS:

    backbone = complex_resnet(backbone_name, in_channels = in_channels, num_classes= num_classes-1, norm_layer=partial(cxnn.BatchNorm2d, naive=False))
    trainable_backbone_layers = 5
    
    if not frozen_backbone is None:
         backbone.load_state_dict(torch.load(frozen_backbone))
         trainable_backbone_layers = 0
    
    backbone_with_fpn = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], 
        extra_blocks=LastLevelP6P7(256, 256), norm_layer=partial(cxnn.BatchNorm2d, naive=False)
        )
    
    model = FCOS(backbone_with_fpn, num_classes, norm_layer=partial(cxnn.BatchNorm2d, naive=False), **kwargs)

    return model