from contrastive_lightning import ContrastiveLightning
from contrastive_dataset import ContrastiveImageDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import torch
import wandb
import argparse

# get config argument
parser = argparse.ArgumentParser(description='Contrastive pretraining.')
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

# initialize datasets
image_train = ContrastiveImageDataset(split=configuration['train_split_name'], dataset_type=configuration['dataset'],
                                      closeness_factor=configuration['closeness_factor'])
image_val = ContrastiveImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                                    closeness_factor=configuration['closeness_factor'])

# initializing dataloaders
train_dataloader = DataLoader(image_train, batch_size=configuration['batch_size'], shuffle=True)
val_dataloader = DataLoader(image_val, batch_size=configuration['batch_size'], shuffle=False)

# training
lightning_module = ContrastiveLightning(learning_rate=configuration['lr'])
trainer = pl.Trainer(max_epochs=configuration['max_epochs'], logger=wandb_logger,
                     callbacks=[checkpoint_callback])
trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
