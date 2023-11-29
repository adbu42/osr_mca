from main_model.u_net_modules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_out_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_out_channels

        self.inc = (Down(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.classifier_output = ClassificationLayer(n_classes)
        self.film = FiLMLayer(n_classes)
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_out_channels))

    def forward(self, x, condition_vector):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        classifier_output = self.classifier_output(x5)
        x_condition = self.film(x5, condition_vector)
        x = self.up1(x_condition, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, classifier_output
