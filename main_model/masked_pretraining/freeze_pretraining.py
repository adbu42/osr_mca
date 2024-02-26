from pytorch_lightning.callbacks import BaseFinetuning
import pytorch_lightning as pl
from torch.optim import Optimizer


class FreezeUnet(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film)
        self.freeze(pl_module.neural_net.classifier_output)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass


class FreezeDenseNet(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film)
        self.freeze(pl_module.neural_net.linear)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass


class FreezeResNet(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film_layer)
        self.freeze(pl_module.neural_net.linear)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass


class FreezeResUNet(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film_layer)
        self.freeze(pl_module.neural_net.linear)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass


class FreezeSimple(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film_layer)
        self.freeze(pl_module.neural_net.linear1)
        self.freeze(pl_module.neural_net.linear2)
        self.freeze(pl_module.neural_net.linear3)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass


class FreezeWideNet(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film_layer)
        self.freeze(pl_module.neural_net.fc)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass
