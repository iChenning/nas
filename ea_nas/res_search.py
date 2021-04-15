import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import OrderedDict
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


def load_normal(load_path):
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def load_supernet(path_):
    state_dict = load_normal(path_)
    new_stage_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'stages.0' in k:
            k = k.replace('stages.0', 'layer1')
        elif 'stages.1' in k:
            k = k.replace('stages.1', 'layer2')
        elif 'stages.2' in k:
            k = k.replace('stages.2', 'layer3')
        elif 'stages.3' in k:
            k = k.replace('stages.3', 'layer4')
        new_stage_dict[k] = v
    return new_stage_dict


import math
# 将resnet的参数映射到supernet中
def remap_res2arch(res, arch):
    arch_tmp = copy.deepcopy(arch)
    arch = copy.deepcopy(res)
    arch.layer1 = copy.deepcopy(arch_tmp.layer1)
    arch.layer2 = copy.deepcopy(arch_tmp.layer2)
    arch.layer3 = copy.deepcopy(arch_tmp.layer3)
    arch.layer4 = copy.deepcopy(arch_tmp.layer4)

    res_stages = [len(res.layer1), len(res.layer2), len(res.layer3), len(res.layer4)]
    arch_stages = [len(arch.layer1), len(arch.layer2), len(arch.layer3), len(arch.layer4)]
    for i_stage, v_stage in enumerate(arch_stages):
        res_layers = getattr(res, 'layer' + str(i_stage + 1))
        arch_layers = getattr(arch, 'layer' + str(i_stage + 1))
        for i_layer in range(v_stage):
            a_layer = arch_layers[i_layer]
            if i_layer > res_stages[i_stage] - 1:
                r_layer = res_layers[res_stages[i_stage] - 1]
            else:
                r_layer = res_layers[i_layer]

            if i_layer == 0:  # 小心遗漏
                a_layer.downsample = copy.deepcopy(r_layer.downsample)

            def copy_conv(r, a):
                # 将r的参数复制到a
                a_out, a_in, a_k_1, a_k_2 = a.weight.shape[:4]
                r_out, r_in, r_k_1, r_k_2 = r.weight.shape[:4]
                out_rate = math.ceil(a_out / r_out)
                in_rate = math.ceil(a_in / r_in)
                k_1_rate = math.ceil(a_k_1 / r_k_1)
                k_2_rate = math.ceil(a_k_2 / r_k_2)
                weight_super = r.weight.clone().repeat(out_rate, in_rate, k_1_rate, k_2_rate)
                weight = weight_super[:a_out, :a_in, :a_k_1, :a_k_2]
                return weight

            def copy_bn(r, a):
                # 将r的参数复制到a
                a_0 = a.weight.shape[0]
                r_0 = r.weight.shape[0]
                rate = math.ceil(a_0 / r_0)

                tmp = r.weight.clone().repeat(rate)
                a_weight = tmp[:a_0]

                tmp = r.bias.clone().repeat(rate)
                a_bias = tmp[:a_0]

                tmp = r.running_mean.clone().repeat(rate)
                a_running_mean = tmp[:a_0]

                tmp = r.running_var.clone().repeat(rate)
                a_running_var = tmp[:a_0]

                return (a_weight, a_bias, a_running_mean, a_running_var)

            a_layer.conv1.weight = Parameter(copy_conv(r_layer.conv1, a_layer.conv1))
            a_weight, a_bias, a_running_mean, a_running_var = copy_bn(r_layer.bn1, a_layer.bn1)
            a_layer.bn1.weight = Parameter(a_weight)
            a_layer.bn1.bias = Parameter(a_bias)
            a_layer.bn1.running_mean = a_running_mean
            a_layer.bn1.running_var = a_running_var

            a_layer.conv2.weight = Parameter(copy_conv(r_layer.conv2, a_layer.conv2))
            a_weight, a_bias, a_running_mean, a_running_var = copy_bn(r_layer.bn2, a_layer.bn2)
            a_layer.bn2.weight = Parameter(a_weight)
            a_layer.bn2.bias = Parameter(a_bias)
            a_layer.bn2.running_mean = a_running_mean
            a_layer.bn2.running_var = a_running_var

            a_layer.conv3.weight = Parameter(copy_conv(r_layer.conv3, a_layer.conv3))
            a_weight, a_bias, a_running_mean, a_running_var = copy_bn(r_layer.bn3, a_layer.bn3)
            a_layer.bn3.weight = Parameter(a_weight)
            a_layer.bn3.bias = Parameter(a_bias)
            a_layer.bn3.running_mean = a_running_mean
            a_layer.bn3.running_var = a_running_var

    return arch


if __name__ == '__main__':
    supernet_stages = (tuple(range(6, 5, -1)),
                       tuple(range(8, 7, -1)),
                       tuple(range(12, 11, -1)),
                       tuple(range(6, 5, -1)))
    supernet_channel_scales = (1.5,)
    supernet_kernels = [(3, 5)]
    supernet_choose = choose_rand(supernet_stages, supernet_channel_scales, supernet_kernels)
    print(supernet_choose[0][1][0])
    supernet = resnet_nas(supernet_choose)

    stages = (tuple(range(6, 0, -1)),
              tuple(range(8, 0, -1)),
              tuple(range(12, 0, -1)),
              tuple(range(6, 0, -1)))
    channel_scales = (1.5, 1.25, 1., 0.75, 0.5, 0.25)
    kernels = [(3, 5), (1, 3)]
    arch_choose = choose_rand(stages, channel_scales, kernels)
    print(arch_choose[0][1][0])
    arch = resnet_nas(arch_choose)
    out_c = arch.stages[0][0].conv1.out_channels
    in_c = arch.stages[0][0].conv1.in_channels
    kernels = arch_choose[0][1][0][1]
    s_weight = supernet.stages[0][0].conv1.weight[:out_c, :in_c, :kernels[0], :kernels[0]]
    a_orig = arch.stages[0][0].conv1.weight[:out_c, :in_c, :kernels[0], :kernels[0]]

    arch = copy.deepcopy(remap_res2arch(supernet, arch))
    a_new = arch.stages[0][0].conv1.weight[:out_c, :in_c, :kernels[0], :kernels[0]]
    print(a_orig.equal(s_weight))
    print(a_new.equal(s_weight))

    data = torch.rand(2, 3, 32, 32)
    out = arch(data)
