from main_model.architectures.wide_net import WideEncoder, BasicBlock
from torch import nn
from torch.functional import F


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, feat_dim=128):
        super(SupConResNet, self).__init__()
        widen_factor = 10
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.encoder = WideEncoder(28, 0.0, nChannels, BasicBlock)
        self.head = nn.Sequential(
            nn.Linear(nChannels[3]*4, nChannels[3]*4),
            nn.ReLU(inplace=True),
            nn.Linear(nChannels[3]*4, feat_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat[0].flatten(1)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
