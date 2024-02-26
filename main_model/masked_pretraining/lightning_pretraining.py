import pytorch_lightning as pl
from main_model.architectures.dense_net_architecture import DenseNet
from main_model.architectures.u_net_architecture import UNet
from main_model.architectures.resnet import ResNet
from main_model.architectures.resunet import ResUNet
from main_model.architectures.simple_architecture import SimpleAutoencoder
from main_model.architectures.wide_net import WideResNet
from torch import optim
import torch.nn as nn
from torchvision.transforms.v2.functional import to_pil_image
import torchvision.transforms.v2 as v2
import torch


class C2AELightning(pl.LightningModule):
    def __init__(self, n_classes, learning_rate=3e-4, architecture='unet', val_dataset=None):
        super().__init__()
        self.n_classes = n_classes
        if architecture == 'unet':
            self.neural_net = UNet(3, 3, n_classes)
        elif architecture == 'densenet':
            self.neural_net = DenseNet(n_classes)
        elif architecture == 'resnet':
            self.neural_net = ResNet(n_classes)
        elif architecture == 'resunet':
            self.neural_net = ResUNet(n_classes)
        elif architecture == 'simple':
            self.neural_net = SimpleAutoencoder(n_classes)
        elif architecture == 'widenet':
            self.neural_net = WideResNet(n_classes)
        else:
            raise ValueError('Architecture not specified correctly!')
        self.n1_loss = nn.L1Loss()
        self.learning_rate = learning_rate
        self.val_dataset = val_dataset
        self.random_erasing = v2.RandomErasing(1, (0.33, 0.33))

    def forward(self, inputs, condition_vector):
        return self.neural_net(inputs, condition_vector)

    def training_step(self, batch, batch_idx):
        full_image, _, _, _ = batch
        masked_image = self.random_erasing(full_image)
        loss, _ = self.reconstruction_step(masked_image, full_image)
        self.log("loss", loss)
        return loss

    def reconstruction_step(self, masked_image, full_image):
        conditional_vector = torch.ones((masked_image.size(0), self.n_classes)).float().cuda()

        model_output, _ = self.neural_net(masked_image, conditional_vector)
        loss = self.n1_loss(model_output, full_image)
        return loss, model_output

    def validation_step(self, batch, batch_idx):
        full_image, _, _, _ = batch
        masked_image = self.random_erasing(full_image)
        loss, reconstruction = self.reconstruction_step(masked_image, full_image)
        self.log("val_loss", loss)

        # log images
        if batch_idx < 3:
            pil_reconstruction = to_pil_image(self.val_dataset.reverse_normalization(reconstruction[0].cpu().detach()))
            pil_masked_image = to_pil_image(self.val_dataset.reverse_normalization(
                masked_image[0].cpu().detach()))
            self.logger.log_image(key=f'images_{batch_idx}',
                                  images=[self.val_dataset.reverse_normalization(full_image[0]), pil_masked_image,
                                          pil_reconstruction],
                                  caption=['input_no_mask', 'masked_image', 'model_reconstruction'])

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=float(self.learning_rate))
        return optimizer
