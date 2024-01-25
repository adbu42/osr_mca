import torch.nn as nn
import torch.nn.functional as F
from main_model.resnet import Bottleneck, Up, ResnetFiLMLayer
import torch


class Concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concat, self).__init__()
        self.concat_convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, encoder_x):
        out = torch.cat([x, encoder_x], dim=1)
        return self.concat_convolution(out)



class ResUNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResUNet, self).__init__()
        self.in_planes = 64

        self.in_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.linear = nn.Linear(512*Bottleneck.expansion, num_classes)
        self.up_layer1 = self._make_layer(Up, 2048, 3, stride=2)
        self.concat1 = Concat(2048, 1024)
        self.up_layer2 = self._make_layer(Up, 1024, 6, stride=2)
        self.concat2 = Concat(1024, 512)
        self.up_layer3 = self._make_layer(Up, 512, 4, stride=2)
        self.concat3 = Concat(512, 256)
        self.up_layer4 = self._make_layer(Up, 128, 3, stride=2)
        self.concat4 = Concat(128, 64)
        self.out_conv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.film_layer = ResnetFiLMLayer(num_classes)
        self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x, condition_vector):
        in1 = F.relu(self.bn1(self.in_conv(x)))
        in2 = self.layer1(in1)
        in3 = self.layer2(in2)
        in4 = self.layer3(in3)
        in5 = self.layer4(in4)
        out = F.avg_pool2d(in5, 4)
        classification = out.view(out.size(0), -1)
        classification = self.linear(classification)
        out = self.film_layer(out, condition_vector)
        out = F.interpolate(out, scale_factor=4, mode='bilinear')
        out = self.up_layer1(out)
        out = self.concat1(out, in4)
        out = self.up_layer2(out)
        out = self.concat2(out, in3)
        out = self.up_layer3(out)
        out = self.concat3(out, in2)
        out = self.up_layer4(out)
        out = self.concat4(out, in1)
        out = self.out_conv(out)
        return self.tanh(out), classification
