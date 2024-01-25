import torch
import torch.nn as nn
import torch.nn.functional as F
from main_model.dense_net_architecture import DecoderTransition, DenseFiLMLayer, EncoderTransition, Bottleneck


class UDecoderTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(UDecoderTransition, self).__init__()
        self.bn = nn.InstanceNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.in_up_conv = nn.Conv2d(out_planes*2, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_planes*2)

    def forward(self, x, in_features):
        x = self.conv(F.relu(self.bn(x)))
        x = self.upsample(x)
        out = torch.cat([in_features, x], dim=1)
        out = self.in_up_conv(F.relu(self.bn2(out)))
        return out


class UDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UDenseNet, self).__init__()

        self.in_conv = nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.enc_dense1 = self._make_dense_layers(Bottleneck, 24, 6)
        self.enc_trans1 = EncoderTransition(96, 48)
        self.enc_dense2 = self._make_dense_layers(Bottleneck, 48, 12)
        self.enc_trans2 = EncoderTransition(192, 96)
        self.enc_dense3 = self._make_dense_layers(Bottleneck, 96, 24)
        self.enc_trans3 = EncoderTransition(384, 192)
        self.enc_dense4 = self._make_dense_layers(Bottleneck, 192, 16)

        self.enc_bn = nn.InstanceNorm2d(384)
        self.linear = nn.Linear(384, num_classes)

        self.film = DenseFiLMLayer(num_classes)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.dec_trans4 = UDecoderTransition(576, 96)
        self.dec_dense4 = self._make_dense_layers(Bottleneck, 384, 16)
        self.dec_trans3 = UDecoderTransition(384, 48)
        self.dec_dense3 = self._make_dense_layers(Bottleneck, 96, 24)
        self.dec_trans2 = UDecoderTransition(192, 24)
        self.dec_dense2 = self._make_dense_layers(Bottleneck, 48, 12)
        self.dec_trans1 = UDecoderTransition(96, 24)
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
        x0 = self.maxpool(x)
        x1 = self.enc_trans1(self.enc_dense1(x0))
        x2 = self.enc_trans2(self.enc_dense2(x1))
        x3 = self.enc_trans3(self.enc_dense3(x2))
        x4 = self.enc_dense4(x3)
        x4 = F.avg_pool2d(F.relu(self.enc_bn(x4)), 4)

        # classifier
        classification = x4.view(x.size(0), -1)
        classification = self.linear(classification)

        # decoder
        out = self.film(x4, condition_vector)

        out = self.upsample1(out)
        out = self.dec_trans4(self.dec_dense4(out), x2)
        out = self.dec_trans3(self.dec_dense3(out), x1)
        out = self.dec_trans2(self.dec_dense2(out), x0)
        out = self.dec_trans1(self.dec_dense1(out), x)
        out = self.out_conv(out)
        
        return self.tanh(out), classification
