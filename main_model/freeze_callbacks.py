from pytorch_lightning.callbacks import BaseFinetuning
import pytorch_lightning as pl
from torch.optim import Optimizer


class FreezeEncoderOrDecoder(BaseFinetuning):
    def __init__(self, switch_epoch=5):
        super().__init__()
        self.switch_epoch = switch_epoch

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.unet.film)
        self.freeze(pl_module.unet.up1)
        self.freeze(pl_module.unet.up2)
        self.freeze(pl_module.unet.up3)
        self.freeze(pl_module.unet.up4)
        self.freeze(pl_module.unet.outc)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        if epoch == self.switch_epoch:
            self.freeze(pl_module.unet.inc)
            self.freeze(pl_module.unet.down1)
            self.freeze(pl_module.unet.down2)
            self.freeze(pl_module.unet.down3)
            self.freeze(pl_module.unet.down4)
            self.freeze(pl_module.unet.classifier_output)
            self.unfreeze_and_add_param_group(pl_module.unet.film, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.unet.up1, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.unet.up2, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.unet.up3, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.unet.up4, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.unet.outc, optimizer=optimizer, train_bn=True)
