from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim import Optimizer
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt


class ToyDataset(Dataset):
    def __init__(self):
        class1 = torch.randn(5000, 2)
        class2 = torch.randn(5000, 2)
        class3 = torch.randn(5000, 2)
        class4 = torch.randn(5000, 2)
        class1[:, 0] += 5
        class2[:, 0] += 5
        class3[:, 0] -= 5
        class4[:, 0] -= 5
        class1[:, 1] += 5
        class2[:, 1] -= 5
        class3[:, 1] += 5
        class4[:, 1] -= 5
        self.dataset = []
        for i, data_class in enumerate([class1, class2, class3, class4]):
            for data_point in data_class:
                self.dataset.append((data_point, i))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_point = self.dataset[idx][0]
        label = self.dataset[idx][1]
        label_matched = True
        non_match_idx = 0
        while label_matched:
            non_match_idx = int(np.random.random_integers(0, len(self.dataset)) - 1)
            non_match_label = self.dataset[non_match_idx][1]
            if non_match_label != label:
                label_matched = False
        non_match_data_point = self.dataset[non_match_idx][0]
        return data_point, label, non_match_data_point, non_match_label


class ToyFilm(nn.Module):
    def __init__(self):
        super().__init__()
        self.hadamard = nn.Linear(4, 8)
        self.addition = nn.Linear(4, 8)

    def forward(self, feature_map, condition_vector):
        hadamard_tensor = self.hadamard(condition_vector)
        addition_tensor = self.addition(condition_vector)
        return torch.mul(feature_map, hadamard_tensor) + addition_tensor


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.en1 = nn.Sequential(nn.Linear(2, 6), nn.Sigmoid())
        self.en2 = nn.Sequential(nn.Linear(6, 8), nn.Sigmoid())
        self.dec1 = nn.Sequential(nn.Linear(8, 6), nn.Sigmoid())
        self.dec2 = nn.Sequential(nn.Linear(6, 2))
        self.classifier = nn.Sequential(nn.Linear(8, 6), nn.Sigmoid(), nn.Linear(6, 4))
        self.film = ToyFilm()

    def forward(self, x, condition_vector):
        x = self.en1(x)
        x = self.en2(x)
        classification = self.classifier(x)
        x = self.film(x, condition_vector)
        x = self.dec1(x)
        x = self.dec2(x)
        return x, classification


class ToyLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n_classes = 4
        self.toy_net = ToyNet()
        self.n1_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.alpha = 0.9
        self.learning_rate = 3e-3
        self.switch_epoch = 30

    def forward(self, inputs, condition_vector):
        return self.toy_net(inputs, condition_vector)

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
        _, classification = self.toy_net(x, conditional_vector)
        classification_loss = self.cross_entropy(classification, y)
        accuracy = self.accuracy(classification, y)
        return classification_loss, accuracy

    def reconstruction_step(self, x, y, x_non_match, y_non_match):
        conditional_vector = F.one_hot(y, self.n_classes).float()
        non_match_conditional_vector = F.one_hot(y_non_match, self.n_classes).float()

        conditional_vector[conditional_vector == 0] = -1
        non_match_conditional_vector[non_match_conditional_vector == 0] = -1

        model_output, _ = self.toy_net(x, conditional_vector)
        non_match_model_output, _ = self.toy_net(x, non_match_conditional_vector)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=float(self.learning_rate))
        return optimizer


class FreezeEncoderOrDecoder(BaseFinetuning):
    def __init__(self, switch_epoch=30):
        super().__init__()
        self.switch_epoch = switch_epoch

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.toy_net.film)
        self.freeze(pl_module.toy_net.dec1)
        self.freeze(pl_module.toy_net.dec2)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        if epoch == self.switch_epoch:
            self.freeze(pl_module.toy_net.en1)
            self.freeze(pl_module.toy_net.en2)
            self.freeze(pl_module.toy_net.classifier)
            self.unfreeze_and_add_param_group(pl_module.toy_net.film, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.toy_net.dec1, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.toy_net.dec2, optimizer=optimizer, train_bn=True)
