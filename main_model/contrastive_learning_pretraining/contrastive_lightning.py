import pytorch_lightning as pl
from contrastive_architecture import SupConResNet
from torch import optim
from contrastive_loss_function import MultiPosConLoss
import torch


class ContrastiveLightning(pl.LightningModule):
    def __init__(self, learning_rate=3e-4):
        super().__init__()
        self.neural_net = SupConResNet()
        self.loss = MultiPosConLoss()
        self.learning_rate = learning_rate

    def forward(self, inputs, condition_vector):
        return self.neural_net(inputs, condition_vector)

    def training_step(self, batch, batch_idx):
        train_images, train_labels = batch
        images = torch.cat([train_images[0], train_images[1]], dim=0)
        labels = torch.cat([train_labels[0], train_labels[1]], dim=0)
        features = self.neural_net(images)
        loss = self.loss(features, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch
        images = torch.cat([val_images[0], val_images[1]], dim=0)
        labels = torch.cat([val_labels[0], val_labels[1]], dim=0)
        features = self.neural_net(images)
        loss = self.loss(features, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        return optimizer
