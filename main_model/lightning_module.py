import pytorch_lightning as pl
import torch

from main_model.architectures.dense_net_architecture import DenseNet
from main_model.architectures.u_net_architecture import UNet
from main_model.architectures.resnet import ResNet
from main_model.architectures.resunet import ResUNet
from main_model.architectures.simple_architecture import SimpleAutoencoder
from main_model.architectures.wide_net import WideResNet
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2.functional import to_pil_image
from torchmetrics.classification import MulticlassAccuracy


class C2AELightning(pl.LightningModule):
    def __init__(self, n_classes, alpha=0.5, learning_rate=3e-4, switch_epoch=5, architecture='unet', val_dataset=None,
                 pretraining_checkpoint=None):
        super().__init__()
        self.n_classes = n_classes
        if architecture == 'unet':
            self.neural_net = UNet(n_classes)
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
        if pretraining_checkpoint is not None:
            checkpoint = torch.load(pretraining_checkpoint)
            encoder_weights = {k[19:]: v for k, v in checkpoint["state_dict"].items()
                               if k.startswith("neural_net.encoder.")}
            self.neural_net.encoder.load_state_dict(encoder_weights)
        self.n1_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.switch_epoch = switch_epoch
        self.val_dataset = val_dataset

    def forward(self, inputs, condition_vector):
        return self.neural_net(inputs, condition_vector)

    def training_step(self, batch, batch_idx):
        x, y, x_non_match, y_non_match = batch
        if self.current_epoch < self.switch_epoch:
            classification_loss, accuracy = self.classification_step(x, y)
            self.log("classification_loss", classification_loss)
            self.log("accuracy", accuracy)
            return classification_loss
        else:
            overall_loss, condition_difference, match_loss, non_match_loss, _, _ = self.reconstruction_step(x, y, x_non_match, y_non_match)
            self.log("match_loss", match_loss)
            self.log("non_match_loss", non_match_loss)
            self.log("overall_loss", overall_loss)
            self.log("condition_difference", condition_difference)
            return overall_loss

    def classification_step(self, x, y):
        conditional_vector = F.one_hot(y, self.n_classes).float()
        _, classification = self.neural_net(x, conditional_vector)
        classification_loss = self.cross_entropy(classification, y)
        accuracy = self.accuracy(classification, y)
        return classification_loss, accuracy

    def reconstruction_step(self, x, y, x_non_match, y_non_match):
        conditional_vector = F.one_hot(y, self.n_classes).float()
        non_match_conditional_vector = F.one_hot(y_non_match, self.n_classes).float()

        conditional_vector[conditional_vector == 0] = -1
        non_match_conditional_vector[non_match_conditional_vector == 0] = -1

        model_output, _ = self.neural_net(x, conditional_vector)
        non_match_model_output, _ = self.neural_net(x, non_match_conditional_vector)
        match_loss = self.n1_loss(model_output, x)
        non_match_loss = self.n1_loss(non_match_model_output, x_non_match)
        overall_loss = self.alpha * match_loss + (1 - self.alpha) * non_match_loss
        non_match_self = self.n1_loss(non_match_model_output, x)
        condition_difference = abs(match_loss - non_match_self)
        return overall_loss, condition_difference, match_loss, non_match_loss, model_output, non_match_model_output

    def validation_step(self, batch, batch_idx):
        x, y, x_non_match, y_non_match = batch
        if self.current_epoch < self.switch_epoch:
            classification_loss, accuracy = self.classification_step(x, y)
            self.log("val_classification_loss", classification_loss)
            self.log("val_accuracy", accuracy)
        else:
            overall_loss, condition_difference, match_loss, non_match_loss, model_output, non_match_model_output = self.reconstruction_step(x, y, x_non_match, y_non_match)
            self.log("val_match_loss", match_loss)
            self.log("val_non_match_loss", non_match_loss)
            self.log("val_overall_loss", overall_loss)
            self.log("val_condition_difference", condition_difference)

            # log images
            if batch_idx < 3:
                pil_match = to_pil_image(self.val_dataset.reverse_normalization(model_output[0].cpu().detach()))
                pil_non_match = to_pil_image(self.val_dataset.reverse_normalization(
                    non_match_model_output[0].cpu().detach()))
                self.logger.log_image(key=f'images_{batch_idx}',
                                      images=[self.val_dataset.reverse_normalization(x[0]), pil_match, pil_non_match],
                                      caption=['input', 'match', 'non_match'])

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=float(self.learning_rate))
        return optimizer
