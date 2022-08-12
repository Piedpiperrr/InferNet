from collections import OrderedDict
import torch
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn


class DSConv (nn.Module):

    def __init__(self, f_3x3, f_1x1, stride=1, padding=0):
        super(DSConv, self).__init__()

        self.feature = nn.Sequential(OrderedDict([
            ('dconv', nn.Conv2d(f_3x3,
                                f_3x3,
                                kernel_size=3,
                                groups=f_3x3,
                                stride=stride,
                                padding=padding,
                                bias=False
                                )),
            ('bn1', nn.BatchNorm2d(f_3x3)),
            ('act1', nn.ReLU()),
            ('pconv', nn.Conv2d(f_3x3,
                                f_1x1,
                                kernel_size=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(f_1x1)),
            ('act2', nn.ReLU())
        ]))

    def forward(self, x):
        out = self.feature(x)
        return out


class InferNet(nn.Module):
    """
        MobileNet-V1 architecture
    """

    def __init__(self, channels, width_multiplier=1.0, num_classes=784):
        super(InferNet, self).__init__()

        channels = [int(elt * width_multiplier) for elt in channels]

        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1, channels[0], kernel_size=3,
                               stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(channels[0])),
            ('act', nn.ReLU())
        ]))

        self.features = nn.Sequential(OrderedDict([
            ('dsconv1', DSConv(channels[0], channels[1], 1, 1)),
            ('dsconv2', DSConv(channels[1], channels[2], 2, 1)),
            ('dsconv3', DSConv(channels[2], channels[2], 1, 1)),
            ('dsconv4', DSConv(channels[2], channels[3], 2, 1)),
            ('dsconv5', DSConv(channels[3], channels[3], 1, 1)),
            ('dsconv6', DSConv(channels[3], channels[4], 2, 1)),
            ('dsconv7_a', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_b', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_c', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_d', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_e', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv8', DSConv(channels[4], channels[5], 2, 1)),
            ('dsconv9', DSConv(channels[5], channels[5], 1, 1))
        ]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels[5], num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv(x)
        out = self.features(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.linear(out)
        out = out.view(-1, 1)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_net():
    return InferNet(
        channels=[
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024],
        width_multiplier=1.0).to(device)