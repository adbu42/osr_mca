from main_model.architectures.resunet import ResUNet
import torch

net = ResUNet()
out, classification = net(torch.randn(2, 3, 64, 64), torch.randn(2, 10))
print(out.size())
print(classification.size())
