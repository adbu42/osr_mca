import torch
import torch.nn as nn
import torch.nn.functional as F
from main_model.architectures.u_net_modules import Up


class WideFiLMLayer(nn.Module):
    def __init__(self, number_known_classes, widen_factor):
        super().__init__()
        self.widen_factor = widen_factor
        self.conditional_vector_encoder_addition = nn.Linear(number_known_classes, 64*self.widen_factor*4)
        self.conditional_vector_encoder_hadamard = nn.Linear(number_known_classes, 64*self.widen_factor*4)

    def forward(self, feature_map, condition_vector):
        hadamard_tensor = (self.conditional_vector_encoder_hadamard(condition_vector)
                           .reshape((condition_vector.size(0), 64*self.widen_factor, 2, 2)))
        added_tensor = (self.conditional_vector_encoder_addition(condition_vector)
                        .reshape((condition_vector.size(0), 64*self.widen_factor, 2, 2)))
        output = torch.mul(feature_map, hadamard_tensor) + added_tensor
        return output


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideEncoder(nn.Module):
    def __init__(self, depth, drop_rate, nChannels, block):
        super(WideEncoder, self).__init__()
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.block1(out1)
        out3 = self.block2(out2)
        out4 = self.block3(out3)
        out = self.relu(self.bn1(out4))
        out = F.avg_pool2d(out, 4)
        return out, out1, out2, out3, out4


class WideClassifier(nn.Module):
    def __init__(self, num_classes, nChannels):
        super(WideClassifier, self).__init__()

        self.fc = nn.Linear(nChannels[3]*4, num_classes)

    def forward(self, x):
        classification = x.flatten(1)
        return self.fc(classification)


class WideDecoder(nn.Module):
    def __init__(self, num_classes, widen_factor):
        super(WideDecoder, self).__init__()
        self.up1 = (Up(64 * widen_factor * 2, 32 * widen_factor))
        self.up2 = (Up(32 * widen_factor * 2, 16 * widen_factor))
        self.up3 = (Up(16 * widen_factor * 2, 16, stride=1))
        self.upsampling = nn.Upsample(scale_factor=4, mode='bilinear')
        self.film_layer = WideFiLMLayer(num_classes, widen_factor)
        self.outc = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        #self.tanh = nn.Tanh()

    def forward(self, x, condition_vector):
        out = self.film_layer(x[0], condition_vector)
        out = self.upsampling(out)
        out = self.up1(out, x[4])
        out = self.up2(out, x[3])
        out = self.up3(out, x[2])
        out = self.outc(torch.cat([x[1], out], dim=1))
        return out


class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        depth = 28
        drop_rate = 0.0
        widen_factor = 10
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        block = BasicBlock
        self.encoder = WideEncoder(depth, drop_rate, nChannels, block)
        self.decoder = WideDecoder(num_classes, widen_factor)
        self.classifier = WideClassifier(num_classes, nChannels)

    def forward(self, x, condition_vector):
        feature_vector = self.encoder(x)
        classification = self.classifier(feature_vector[0])
        out = self.decoder(feature_vector, condition_vector)
        return out, classification
