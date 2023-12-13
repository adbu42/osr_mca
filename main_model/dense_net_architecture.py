import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class EncoderTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(EncoderTransition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DecoderTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DecoderTransition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.upsample(out)
        return out


class DenseFiLMLayer(nn.Module):
    def __init__(self, number_known_classes):
        super().__init__()
        self.conditional_vector_encoder_addition = nn.Linear(number_known_classes, 384)
        self.conditional_vector_encoder_hadamard = nn.Linear(number_known_classes, 384)
        self.batch_norm = nn.BatchNorm2d(384)

    def forward(self, feature_map, condition_vector):
        hadamard_tensor = (self.conditional_vector_encoder_hadamard(condition_vector)
                           .reshape((condition_vector.size(0), 384, 1, 1)))
        added_tensor = (self.conditional_vector_encoder_addition(condition_vector)
                        .reshape((condition_vector.size(0), 384, 1, 1)))
        output = torch.mul(feature_map, hadamard_tensor) + added_tensor
        return self.batch_norm(output)


class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()

        self.in_conv = nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.enc_dense1 = self._make_dense_layers(Bottleneck, 24, 6)
        self.enc_trans1 = EncoderTransition(96, 48)
        self.enc_dense2 = self._make_dense_layers(Bottleneck, 48, 12)
        self.enc_trans2 = EncoderTransition(192, 96)
        self.enc_dense3 = self._make_dense_layers(Bottleneck, 96, 24)
        self.enc_trans3 = EncoderTransition(384, 192)
        self.enc_dense4 = self._make_dense_layers(Bottleneck, 192, 16)

        self.enc_bn = nn.BatchNorm2d(384)
        self.linear = nn.Linear(384, num_classes)

        self.film = DenseFiLMLayer(num_classes)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.dec_trans4 = DecoderTransition(576, 96)
        self.dec_dense4 = self._make_dense_layers(Bottleneck, 384, 16)
        self.dec_trans3 = DecoderTransition(384, 48)
        self.dec_dense3 = self._make_dense_layers(Bottleneck, 96, 24)
        self.dec_trans2 = DecoderTransition(192, 24)
        self.dec_dense2 = self._make_dense_layers(Bottleneck, 48, 12)
        self.dec_trans1 = DecoderTransition(96, 24)
        self.dec_dense1 = self._make_dense_layers(Bottleneck, 24, 6)

        self.out_conv = nn.Conv2d(24, 3, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, 12))
            in_planes += 12
        return nn.Sequential(*layers)

    def forward(self, x, condition_vector):
        # encoder
        x = self.in_conv(x)
        x = self.maxpool(x)
        x = self.enc_trans1(self.enc_dense1(x))
        x = self.enc_trans2(self.enc_dense2(x))
        x = self.enc_trans3(self.enc_dense3(x))
        x = self.enc_dense4(x)
        x = F.avg_pool2d(F.relu(self.enc_bn(x)), 4)

        # classifier
        classification = x.view(x.size(0), -1)
        classification = self.linear(classification)

        # decoder
        out = self.film(x, condition_vector)

        out = self.upsample1(out)
        out = self.dec_trans4(self.dec_dense4(out))
        out = self.dec_trans3(self.dec_dense3(out))
        out = self.dec_trans2(self.dec_dense2(out))
        out = self.dec_trans1(self.dec_dense1(out))
        out = self.out_conv(out)
        
        return self.tanh(out), classification
