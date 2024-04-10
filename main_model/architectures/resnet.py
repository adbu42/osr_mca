import torch.nn as nn
import torch.nn.functional as F
import torch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Up(nn.Module):
    expansion = 1/2

    def __init__(self, in_planes, planes, stride=1):
        super(Up, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        output_padding = stride-1
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3,
                                        stride=stride, padding=1,  output_padding=output_padding,bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, int(self.expansion * planes), kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(self.expansion*planes), track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != int(self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, int(self.expansion*planes),
                          kernel_size=1, stride=stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(int(self.expansion*planes), track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResnetFiLMLayer(nn.Module):
    def __init__(self, number_known_classes):
        super().__init__()
        self.conditional_vector_encoder_addition = nn.Linear(number_known_classes, 2048)
        self.conditional_vector_encoder_hadamard = nn.Linear(number_known_classes, 2048)

    def forward(self, feature_map, condition_vector):
        hadamard_tensor = (self.conditional_vector_encoder_hadamard(condition_vector)
                           .reshape((condition_vector.size(0), 2048, 1, 1)))
        added_tensor = (self.conditional_vector_encoder_addition(condition_vector)
                        .reshape((condition_vector.size(0), 2048, 1, 1)))
        output = torch.mul(feature_map, hadamard_tensor) + added_tensor
        return output


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.in_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.linear = nn.Linear(512*Bottleneck.expansion, num_classes)
        self.up_layer1 = self._make_layer(Up, 2048, 3, stride=2)
        self.up_layer2 = self._make_layer(Up, 1024, 6, stride=2)
        self.up_layer3 = self._make_layer(Up, 512, 4, stride=2)
        self.up_layer4 = self._make_layer(Up, 128, 3, stride=2)
        self.out_conv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.film_layer = ResnetFiLMLayer(num_classes)
        #self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x, condition_vector):
        out = F.relu(self.bn1(self.in_conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        classification = out.view(out.size(0), -1)
        classification = self.linear(classification)
        out = self.film_layer(out, condition_vector)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up_layer1(out)
        out = self.up_layer2(out)
        out = self.up_layer3(out)
        out = self.up_layer4(out)
        out = self.out_conv(out)
        return out, classification
