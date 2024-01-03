from pytorch_lightning.callbacks import BaseFinetuning
import pytorch_lightning as pl
from torch.optim import Optimizer


class FreezeUnet(BaseFinetuning):
    def __init__(self, switch_epoch=5):
        super().__init__()
        self.switch_epoch = switch_epoch

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film)
        self.freeze(pl_module.neural_net.up1)
        self.freeze(pl_module.neural_net.up2)
        self.freeze(pl_module.neural_net.up3)
        self.freeze(pl_module.neural_net.up4)
        self.freeze(pl_module.neural_net.outc)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        if epoch == self.switch_epoch:
            self.freeze(pl_module.neural_net.inc)
            self.freeze(pl_module.neural_net.down1)
            self.freeze(pl_module.neural_net.down2)
            self.freeze(pl_module.neural_net.down3)
            self.freeze(pl_module.neural_net.down4)
            self.freeze(pl_module.neural_net.classifier_output)
            self.unfreeze_and_add_param_group(pl_module.neural_net.film, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.up1, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.up2, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.up3, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.up4, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.outc, optimizer=optimizer, train_bn=True)


class FreezeDenseNet(BaseFinetuning):
    def __init__(self, switch_epoch=5):
        super().__init__()
        self.switch_epoch = switch_epoch

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.neural_net.film)
        self.freeze(pl_module.neural_net.dec_trans4)
        self.freeze(pl_module.neural_net.dec_dense4)
        self.freeze(pl_module.neural_net.dec_trans3)
        self.freeze(pl_module.neural_net.dec_dense3)
        self.freeze(pl_module.neural_net.dec_trans2)
        self.freeze(pl_module.neural_net.dec_dense2)
        self.freeze(pl_module.neural_net.dec_trans1)
        self.freeze(pl_module.neural_net.dec_dense1)
        self.freeze(pl_module.neural_net.out_conv)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        if epoch == self.switch_epoch:
            self.freeze(pl_module.neural_net.in_conv)
            self.freeze(pl_module.neural_net.enc_dense1)
            self.freeze(pl_module.neural_net.enc_trans1)
            self.freeze(pl_module.neural_net.enc_dense2)
            self.freeze(pl_module.neural_net.enc_trans2)
            self.freeze(pl_module.neural_net.enc_dense3)
            self.freeze(pl_module.neural_net.enc_trans3)
            self.freeze(pl_module.neural_net.enc_dense4)
            self.freeze(pl_module.neural_net.linear)

            self.unfreeze_and_add_param_group(pl_module.neural_net.film, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_trans4, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_dense4, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_trans3, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_dense3, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_trans2, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_dense2, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_trans1, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.dec_dense1, optimizer=optimizer, train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.neural_net.out_conv, optimizer=optimizer, train_bn=True)
