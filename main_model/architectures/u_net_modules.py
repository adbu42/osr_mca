import torch
import torch.nn as nn
from torch.nn import functional as F

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        self.instance_norm = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        return F.leaky_relu(self.instance_norm(self.conv(x)), 0.2)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        padding = 1 if stride == 2 else 1
        kernel_size = 4 if stride == 2 else 3

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, track_running_stats=False)
        )

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return F.relu(self.up(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.up(x)


class FiLMLayer(nn.Module):
    def __init__(self, number_known_classes):
        super().__init__()
        self.conditional_vector_encoder_addition = nn.Linear(number_known_classes, 4096)
        self.conditional_vector_encoder_hadamard = nn.Linear(number_known_classes, 4096)

    def forward(self, feature_map, condition_vector):
        hadamard_tensor = (self.conditional_vector_encoder_hadamard(condition_vector)
                           .reshape((condition_vector.size(0), 1024, 2, 2)))
        added_tensor = (self.conditional_vector_encoder_addition(condition_vector)
                        .reshape((condition_vector.size(0), 1024, 2, 2)))
        conditioned_feature_map = torch.mul(feature_map, hadamard_tensor) + added_tensor
        return conditioned_feature_map


class ClassificationLayer(nn.Module):
    def __init__(self, number_known_classes):
        super().__init__()
        self.classifier_output = nn.Linear(4096, number_known_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.classifier_output(x)


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.instance_norm = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        return F.leaky_relu(self.instance_norm(self.conv(x)), 0.2)
