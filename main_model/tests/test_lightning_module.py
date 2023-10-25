import wandb

from main_model.lightning_module import C2AELightning
from main_model.tiny_image_dataset import TinyImageDataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import yaml
import torch

# optimize torch for cuda
torch.set_float32_matmul_precision('high')

# initialize wandb
with open('../config.yml', 'r') as file:
    configuration = yaml.safe_load(file)
wandb.login(key=configuration['wandb_api_key'])
wandb_logger = WandbLogger(project='test_u_net')

tiny_image_train = TinyImageDataset(split='train')
tiny_image_test = TinyImageDataset(split='valid')
c2ae = C2AELightning()


# initializing dataloaders
train_dataloader = DataLoader(tiny_image_train, batch_size=20, shuffle=True)
test_dataloader = DataLoader(tiny_image_test, batch_size=20, shuffle=False)

# training
trainer = pl.Trainer(limit_train_batches=200, max_epochs=1, logger=wandb_logger)
trainer.fit(model=c2ae, train_dataloaders=train_dataloader)

# testing
trainer.test(model=c2ae, dataloaders=test_dataloader)
