import lightning.pytorch as pl
from main_model.u_net_architecture import UNet
from torch import optim
import torch.nn as nn


class C2AELightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3, 3)

    def forward(self, inputs):
        return self.unet(inputs)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        model_output = self.unet(x)
        loss = nn.functional.mse_loss(model_output, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        model_output = self.unet(x)
        loss = nn.functional.mse_loss(model_output, x)
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
