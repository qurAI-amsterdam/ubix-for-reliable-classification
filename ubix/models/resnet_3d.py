import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    "ResNet",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def add_dropout(config, layer):
    if config.drop_rate > 0:
        return nn.Sequential(layer, nn.Dropout(p=config.drop_rate))
    else:
        return layer


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


def get_norm(config, num_features):
    if config.network_norm_type == "batch_norm":
        return nn.BatchNorm3d(num_features)
    elif config.network_norm_type == "group_norm":
        return nn.GroupNorm(32, num_features)
    elif config.network_norm_type == "instance_norm":
        return nn.InstanceNorm3d(num_features)
    else:
        raise ValueError(
            f"config.network_norm_type {config.network_norm_type} unknown."
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, config, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = get_norm(config, planes)
        self.relu = nn.ReLU(inplace=True)
        self.relu = add_dropout(config, self.relu)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = get_norm(config, planes)
        self.bn2 = add_dropout(config, self.bn2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, config, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(config, planes)
        self.bn1 = add_dropout(config, self.bn1)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = get_norm(config, planes)
        self.bn2 = add_dropout(config, self.bn2)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = get_norm(config, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.relu = add_dropout(config, self.relu)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        shortcut_type="B",
        num_classes=400,
        input_channels=3,
        config=None,
    ):
        self.config = config
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.conv1 = add_dropout(self.config, self.conv1)

        self.bn1 = get_norm(self.config, 64)
        self.relu = nn.ReLU(inplace=True)
        if config.maxpool_z:
            self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool3d(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            )
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = lambda shape: nn.AvgPool3d(shape, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if config.devries_confidence:
            self.fc_devries = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    get_norm(self.config, planes * block.expansion),
                )
                downsample = add_dropout(self.config, downsample)

        layers = []
        layers.append(block(self.config, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.config, self.inplanes, planes))

        return nn.Sequential(*layers)

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def forward(self, x, return_features=False, return_devries_confidence=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(shape=x.shape[-3:])(x)

        x = x.view(x.size(0), -1)

        if return_features:
            return x

        pred = self.fc(x)

        if return_devries_confidence and self.config.devries_confidence:
            confidence = self.fc_devries(x)
            return pred, confidence

        return pred


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append("layer{}".format(i))
    ft_module_names.append("fc")

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({"params": v})
                break
        else:
            parameters.append({"params": v, "lr": 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
