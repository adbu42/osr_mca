from main_model.architectures.u_net_modules import *


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()

        self.inc = (Down(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.classifier_output = ClassificationLayer(n_classes)
        self.film = FiLMLayer(n_classes)
        self.up1 = (Up(2048, 512))
        self.up2 = (Up(1024, 256))
        self.up3 = (Up(512, 128))
        self.up4 = (Up(256, 64))
        self.outc = (OutConv(128, 3))

    def forward(self, x, condition_vector):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        classifier_output = self.classifier_output(x5)
        x_condition = self.film(x5, condition_vector)
        x = self.up1(x_condition, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        logits = self.outc(x, x1)
        return logits, classifier_output
