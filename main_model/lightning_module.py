import pytorch_lightning as pl
from main_model.u_net_architecture import UNet
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchmetrics.classification import MulticlassAccuracy


class C2AELightning(pl.LightningModule):
    def __init__(self, n_classes, alpha=0.5, learning_rate=3e-4, switch_epoch=5):
        super().__init__()
        self.n_classes = n_classes
        self.unet = UNet(3, 3, self.n_classes)
        self.n1_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.switch_epoch = switch_epoch

    def forward(self, inputs, condition_vector):
        return self.unet(inputs, condition_vector)

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
        _, classification = self.unet(x, conditional_vector)
        classification_loss = self.cross_entropy(classification, y)
        accuracy = self.accuracy(classification, y)
        return classification_loss, accuracy

    def reconstruction_step(self, x, y, x_non_match, y_non_match):
        conditional_vector = F.one_hot(y, self.n_classes).float()
        non_match_conditional_vector = F.one_hot(y_non_match, self.n_classes).float()

        conditional_vector[conditional_vector == 0] = -1
        non_match_conditional_vector[non_match_conditional_vector == 0] = -1

        model_output, _ = self.unet(x, conditional_vector)
        non_match_model_output, _ = self.unet(x, non_match_conditional_vector)
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
                pil_match = to_pil_image(model_output[0].cpu().detach())
                pil_non_match = to_pil_image(non_match_model_output[0].cpu().detach())
                self.logger.log_image(key=f'images_{batch_idx}', images=[x[0], pil_match, pil_non_match],
                                      caption=['input', 'match', 'non_match'])

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=float(self.learning_rate))
        return optimizer
