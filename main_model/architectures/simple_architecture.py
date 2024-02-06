import torch
from torch import nn
from torch.nn import functional as F


#class SimpleFiLMLayer(nn.Module):
#    def __init__(self, number_known_classes):
#        super().__init__()
#        self.conditional_vector_encoder_addition = nn.Linear(number_known_classes, 12800)
#        self.conditional_vector_encoder_hadamard = nn.Linear(number_known_classes, 12800)
#        self.conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

#    def forward(self, feature_map, condition_vector):
#        hadamard_tensor = (self.conditional_vector_encoder_hadamard(condition_vector)
#                           .reshape((condition_vector.size(0), 512, 5, 5)))
#        added_tensor = (self.conditional_vector_encoder_addition(condition_vector)
#                        .reshape((condition_vector.size(0), 512, 5, 5)))
#        conditioned_feature_map = torch.mul(feature_map, hadamard_tensor) + added_tensor
#        output = torch.cat([conditioned_feature_map, feature_map], dim=1)
#        return self.conv(output)


class SimpleFiLMLayer(nn.Module):
    def __init__(self, number_known_classes):
        super().__init__()
        self.conditional_vector_encoder_addition = nn.Linear(number_known_classes, 12800)
        self.conditional_vector_encoder_hadamard = nn.Linear(number_known_classes, 12800)
        #self.conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

    def forward(self, feature_map, condition_vector):
        hadamard_tensor = (self.conditional_vector_encoder_hadamard(condition_vector)
                           .reshape((condition_vector.size(0), 512, 5, 5)))
        added_tensor = (self.conditional_vector_encoder_addition(condition_vector)
                        .reshape((condition_vector.size(0), 512, 5, 5)))
        conditioned_feature_map = torch.mul(feature_map, hadamard_tensor) + added_tensor
        #output = torch.cat([conditioned_feature_map, feature_map], dim=1)
        return conditioned_feature_map#self.conv(output)




class SimpleAutoencoder(nn.Module):
    def __init__(self, n_classes):
        super(SimpleAutoencoder, self).__init__()
        d = 128
        self.deconv1 = nn.ConvTranspose2d(512, d * 2, 4, 1, 0)
        self.deconv1_bn = nn.InstanceNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 2, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.InstanceNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.InstanceNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        self.film_layer = SimpleFiLMLayer(n_classes)
        self.conv1 = nn.Conv2d(3, d // 2, 4, 2, 1)
        self.conv1_bn = nn.InstanceNorm2d(d // 2)
        self.conv2 = nn.Conv2d(d // 2, d * 2, 4, 2, 1)
        self.conv2_bn = nn.InstanceNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.InstanceNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 512, 4, 1, 0)
        self.conv4_bn = nn.InstanceNorm2d(d * 4)

        self.linear1 = nn.Linear(12800, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, n_classes)

    def encode(self, x):
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        out = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        return out

    def decode(self, z):
        x = F.relu(self.deconv1_bn(self.deconv1(z)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x

    def forward(self, x, condition_vector):
        intermediate = self.encode(x)
        classification = F.leaky_relu(self.linear1(intermediate.reshape(condition_vector.size(0), 12800)), 0.2)
        classification = F.leaky_relu((self.linear2(classification)), 0.2)
        classification = self.linear3(classification)
        intermediate = self.film_layer(intermediate, condition_vector)
        return self.decode(intermediate), classification
