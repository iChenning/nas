from .vit import *
from .resnet import *
from .wide_resnet import *
"""
    ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']
    ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
    ['wide_resnet28_10']
"""
"""
    Encoder only used to generate feature, it doesn't have fc.
    The feature generated from encoder didn't normalize.
"""


from .fc import *
"""
    ['dot', 'cos']
"""


from .criterion import *
"""
    ['cross_entropy', 'auto_weight']
"""

