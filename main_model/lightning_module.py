import lightning.pytorch as pl
from main_model.u_net_architecture import UNet
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


class C2AELightning(pl.LightningModule):
    def __init__(self, n_classes, alpha=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.unet = UNet(3, 3, self.n_classes)
        self.loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, inputs, condition_vector):
        return self.unet(inputs, condition_vector)

    def training_step(self, batch, batch_idx):
        x, y, x_non_match, y_non_match = batch

        conditional_vector = F.one_hot(y, self.n_classes).float()
        non_match_conditional_vector = F.one_hot(y_non_match, self.n_classes).float()

        conditional_vector[conditional_vector == 0] = -1
        non_match_conditional_vector[non_match_conditional_vector == 0] = -1

        model_output = self.unet(x, conditional_vector)
        non_match_model_output = self.unet(x, non_match_conditional_vector)
        match_loss = self.loss(model_output, x)
        non_match_loss = self.loss(non_match_model_output, x_non_match)
        overall_loss = self.alpha * match_loss + (1 - self.alpha) * non_match_loss

        # logging
        self.log("match_loss", match_loss)
        self.log("non_match_loss", non_match_loss)
        self.log("overall_loss", overall_loss)

        return overall_loss

    def test_step(self, batch, batch_idx):
        x, y, x_non_match, y_non_match = batch

        conditional_vector = F.one_hot(y, self.n_classes).float()
        non_match_conditional_vector = F.one_hot(y_non_match, self.n_classes).float()

        conditional_vector[conditional_vector == 0] = -1
        non_match_conditional_vector[non_match_conditional_vector == 0] = -1

        model_output = self.unet(x, conditional_vector)
        non_match_model_output = self.unet(x, non_match_conditional_vector)
        match_loss = self.loss(model_output, x)
        non_match_loss = self.loss(non_match_model_output, x_non_match)
        overall_loss = self.alpha * match_loss + (1 - self.alpha) * non_match_loss
        non_match_self = self.loss(non_match_model_output, x)

        # logging
        self.log("test_self_non_match_loss", non_match_self)
        self.log("test_match_loss", match_loss)
        self.log("test_non_match_loss", non_match_loss)
        self.log("test_overall_loss", overall_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
