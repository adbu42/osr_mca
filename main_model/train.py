from lightning_module import C2AELightning
from freeze_callbacks import *
from dataset import ImageDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import torch
import wandb
import argparse

# get config argument
parser = argparse.ArgumentParser(description='Train the C2AE model.')
parser.add_argument('--config', help='path to config with common train settings, such as LR')
parsed_args = parser.parse_args()

# optimize torch for cuda
torch.set_float32_matmul_precision('high')

# initialize configurations
with open(parsed_args.config, 'r') as file:
    configuration = yaml.safe_load(file)

with open('api_key.yml', 'r') as file:
    api_key = yaml.safe_load(file)

# initialize wandb
wandb.login(key=api_key['wandb_api_key'])
wandb_logger = WandbLogger(project=configuration['project_name'], save_dir='tests/runs')
wandb_logger.log_hyperparams(configuration)

# initialize callbacks
checkpoint_callback = ModelCheckpoint(every_n_epochs=configuration['checkpoint_epoch_interval'], save_top_k=-1)

if configuration['architecture'] == 'unet':
    freeze_callback = FreezeUnet(switch_epoch=configuration['switch_epoch'])
elif configuration['architecture'] == 'densenet':
    freeze_callback = FreezeDenseNet(switch_epoch=configuration['switch_epoch'])
elif configuration['architecture'] == 'resnet':
    freeze_callback = FreezeResNet(switch_epoch=configuration['switch_epoch'])
elif configuration['architecture'] == 'resunet':
    freeze_callback = FreezeResUNet(switch_epoch=configuration['switch_epoch'])
elif configuration['architecture'] == 'simple':
    freeze_callback = FreezeSimple(switch_epoch=configuration['switch_epoch'])
elif configuration['architecture'] == 'widenet':
    freeze_callback = FreezeWideNet(switch_epoch=configuration['switch_epoch'])
else:
    raise ValueError('Architecture name wrong!')

# initialize datasets
image_train = ImageDataset(split=configuration['train_split_name'], dataset_type=configuration['dataset'],
                           is_close=True, closeness_factor=configuration['closeness_factor'])
image_val = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                         is_close=True, closeness_factor=configuration['closeness_factor'])

# initializing dataloaders
train_dataloader = DataLoader(image_train, batch_size=configuration['batch_size'], shuffle=True)
val_dataloader = DataLoader(image_val, batch_size=configuration['batch_size'], shuffle=False)


# training
c2ae = C2AELightning(image_train.num_classes(), learning_rate=configuration['lr'],
                     switch_epoch=configuration['switch_epoch'], architecture=configuration['architecture'],
                     val_dataset=image_val)
trainer = pl.Trainer(max_epochs=configuration['max_epochs'], logger=wandb_logger,
                     callbacks=[checkpoint_callback, freeze_callback])
trainer.fit(model=c2ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
