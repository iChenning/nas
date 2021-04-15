import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import copy
from torch.nn.parameter import Parameter


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, kernel=(3, 5), stride=1, downsample=None,
                 base_width=64, channel_rate=1.5, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.) * channel_rate)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=kernel[0],
                               stride=1, padding=kernel[0] // 2, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=kernel[1],
                               stride=stride, padding=kernel[1] // 2, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=kernel[0],
                               stride=1, padding=kernel[0] // 2, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, arch_choose, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [v[0] for v in arch_choose]
        self.layer1 = self._make_layer(block, 64, layers[0], arch_choose[0][1])
        self.layer2 = self._make_layer(block, 128, layers[1], arch_choose[1][1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], arch_choose[2][1], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], arch_choose[3][1], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.emb_size = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, layer_choose, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 有下采样的layer
        layers.append(block(self.inplanes, planes, kernel=layer_choose[0][1], stride=stride,
                            downsample=downsample, base_width=self.base_width, channel_rate=layer_choose[0][0]))
        self.inplanes = planes * block.expansion
        # 无下采样的layer
        for i_ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel=layer_choose[i_][1], stride=1,
                                downsample=None, base_width=self.base_width, channel_rate=layer_choose[i_][0]))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def _resnet(arch, block, arch_choose, **kwargs):
    model = ResNet(block, arch_choose, **kwargs)
    return model


def resnet_nas(arch_choose=None, **kwargs):
    return _resnet('resnet50', Bottleneck, arch_choose, **kwargs)


def choose_rand(stages, channel_scales, kernels):
    choose = []
    for stage in stages:
        num_layers = random.choice(stage)
        layer_choose = []
        for i_layer in range(num_layers):
            channel_rate = random.choice(channel_scales)
            kernel = random.choice(kernels)
            layer_choose.append([channel_rate, kernel])
        choose.append([num_layers, layer_choose])
    return choose


# 参数赋值,将supernet的参数映射到arch中
def weight_remap_supernet2arch(supernet, arch, arch_choose):
    arch_tmp = copy.deepcopy(supernet)
    arch_tmp.stages = copy.deepcopy(arch.stages)
    for i_stage, v_stage in enumerate(arch_choose):  # 选择layers（depth）
        num_layers = v_stage[0]
        layers_choose = v_stage[1]
        s_layers = supernet.stages[i_stage]
        a_layers = arch_tmp.stages[i_stage]
        for i_layer in range(num_layers):  # 选择channel和kernel（合起来width）
            s_layer = s_layers[i_layer]
            a_layer = a_layers[i_layer]

            kernel = layers_choose[i_layer][1]
            kernel_conv1 = kernel[0]
            kernel_conv2 = kernel[1]

            if i_layer == 0:
                a_layer.downsample = copy.deepcopy(s_layer.downsample)

            tmp = s_layer.conv1.weight.clone()
            a_layer.conv1.weight = Parameter(tmp[
                                             :a_layer.conv1.out_channels,
                                             :a_layer.conv1.in_channels,
                                             :kernel_conv1,
                                             :kernel_conv1])
            a_layer.bn1.weight = Parameter(s_layer.bn1.weight.clone()[:a_layer.bn1.weight.shape[0]])
            a_layer.bn1.bias = Parameter(s_layer.bn1.bias.clone()[:a_layer.bn1.bias.shape[0]])
            a_layer.bn1.running_mean = s_layer.bn1.running_mean.clone()[:a_layer.bn1.running_mean.shape[0]]
            a_layer.bn1.running_var = s_layer.bn1.running_var.clone()[:a_layer.bn1.running_var.shape[0]]

            tmp = s_layer.conv2.weight.clone()
            a_layer.conv2.weight = Parameter(tmp[
                                             :a_layer.conv2.out_channels,
                                             :a_layer.conv2.in_channels,
                                             :kernel_conv2,
                                             :kernel_conv2])
            a_layer.bn2.weight = Parameter(s_layer.bn2.weight.clone()[:a_layer.bn2.weight.shape[0]])
            a_layer.bn2.bias = Parameter(s_layer.bn2.bias.clone()[:a_layer.bn2.bias.shape[0]])
            a_layer.bn2.running_mean = s_layer.bn2.running_mean.clone()[:a_layer.bn2.running_mean.shape[0]]
            a_layer.bn2.running_var = s_layer.bn2.running_var.clone()[:a_layer.bn2.running_var.shape[0]]

            tmp = s_layer.conv3.weight.clone()
            a_layer.conv3.weight = Parameter(tmp[
                                             :a_layer.conv3.out_channels,
                                             :a_layer.conv3.in_channels,
                                             :kernel_conv1,
                                             :kernel_conv1])
            a_layer.bn3.weight = Parameter(s_layer.bn3.weight.clone())
            a_layer.bn3.bias = Parameter(s_layer.bn3.bias.clone())
            a_layer.bn3.running_mean = s_layer.bn3.running_mean.clone()
            a_layer.bn3.running_var = s_layer.bn3.running_var.clone()

    return arch_tmp


from utils.modules import import_encoder

from thop import profile

if __name__ == '__main__':
    supernet_stages = (tuple(range(6, 5, -1)),
                       tuple(range(8, 7, -1)),
                       tuple(range(12, 11, -1)),
                       tuple(range(6, 5, -1)))
    supernet_cs = (1.5,)
    supernet_kernels = [(3, 5)]
    supernet_choose = choose_rand(supernet_stages, supernet_cs, supernet_kernels)
    supernet = resnet_nas(supernet_choose)

    choose_ = [[3, [[1.0, (1, 3)], [0.25, (1, 3)], [1.25, (3, 5)]]],
               [3, [[1.5, (3, 5)], [1.5, (3, 5)], [0.5, (3, 5)]]],
               [7, [[1.5, (3, 5)], [1.0, (3, 5)], [0.25, (3, 5)], [0.75, (1, 3)], [0.5, (1, 3)], [1.5, (1, 3)],
                    [0.25, (3, 5)]]],
               [4, [[0.75, (3, 5)], [1.5, (1, 3)], [0.25, (3, 5)], [0.25, (3, 5)]]]]
    choose_ = [[3, [[1., (1, 3)], [1., (1, 3)], [1., (1, 3)]]],
               [5, [[1., (1, 3)], [1., (1, 3)], [1., (1, 3)], [1., (1, 3)], [1., (1, 3)]]],
               [6, [[1., (1, 3)], [1., (1, 3)], [1., (1, 3)], [1., (1, 3)], [1., (1, 3)], [1., (1, 3)]]],
               [2, [[1., (1, 3)], [1., (1, 3)]]]]

    # choose_ = [[3, [[1.25, (3, 5)], [0.75, (3, 5)], [0.5, (3, 5)]]],
    #            [7, [[1.25, (3, 5)], [1.5, (1, 3)], [1.25, (3, 5)], [0.75, (3, 5)], [1.0, (1, 3)], [0.5, (3, 5)],
    #                 [0.25, (1, 3)]]],
    #            [5, [[1.5, (3, 5)], [0.75, (3, 5)], [1.0, (1, 3)], [1.0, (1, 3)], [0.75, (1, 3)]]],
    #            [1, [[1.25, (3, 5)]]]]

    arch = resnet_nas(choose_)

    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(arch, inputs=(input,))
    print('macs:', macs / 1e9, ' params:', params / 1e6)
