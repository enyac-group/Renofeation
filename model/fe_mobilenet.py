import torch
from torch import nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 dropout=0,
                 haslinear=False,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        self.haslinear = haslinear
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(dropout)
        else:
            self.dropout_layer = None

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        layers = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if s == 2:
                layers.append(features)
                features = []
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        layers.append(features)

        self.layer1 = nn.Sequential(*layers[0])
        self.layer2 = nn.Sequential(*layers[1])
        self.layer3 = nn.Sequential(*layers[2])
        self.layer4 = nn.Sequential(*layers[3])
        self.layer5 = nn.Sequential(*layers[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.layer1(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.layer2(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.layer3(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.layer4(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.layer5(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.haslinear:
            x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def fembnetv2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(haslinear=False, **kwargs)
    if pretrained:

        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        del state_dict['classifier.1.weight']
        del state_dict['classifier.1.bias']

        new_dict = {}
        cumulative_layer = 0
        for li, layers in enumerate([4, 3, 4, 6, 2]):
            for l in range(layers):
                key = 'features.{}.'.format(l+cumulative_layer)
                new_key = 'layer{}.{}.'.format(li+1, l)

                for k in state_dict:
                    if key in k:
                        new_k = k.replace(key, new_key)
                        new_dict[new_k] = state_dict[k]
            cumulative_layer += layers

        new_params = model.state_dict()
        new_params.update(new_dict)
        model.load_state_dict(new_params)
    return model

def mbnetv2_dropout(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(haslinear=True, **kwargs)
    if pretrained:

        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        del state_dict['classifier.1.weight']
        del state_dict['classifier.1.bias']

        new_dict = {}
        cumulative_layer = 0
        for li, layers in enumerate([4, 3, 4, 6, 2]):
            for l in range(layers):
                key = 'features.{}.'.format(l+cumulative_layer)
                new_key = 'layer{}.{}.'.format(li+1, l)

                for k in state_dict:
                    if key in k:
                        new_k = k.replace(key, new_key)
                        new_dict[new_k] = state_dict[k]
            cumulative_layer += layers

        new_params = model.state_dict()
        new_params.update(new_dict)
        model.load_state_dict(new_params)
    return model
