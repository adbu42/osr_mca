from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
import torch
import wandb


# optimize torch for cuda
torch.set_float32_matmul_precision('high')

# initialize configurations and wandb
with open('config.yml', 'r') as file:
    configuration = yaml.safe_load(file)

with open('api_key.yml', 'r') as file:
    api_key = yaml.safe_load(file)

wandb.login(key=api_key['wandb_api_key'])
wandb_logger = WandbLogger(project=configuration['project_name'], save_dir='tests/runs')
wandb_logger.log_hyperparams(configuration)

# initialize callbacks
checkpoint_callback = ModelCheckpoint(every_n_epochs=configuration['checkpoint_epoch_interval'],
                                      filename='c2ae-{epoch:02d}-{condition_difference:.2f}',
                                      monitor='condition_difference',
                                      mode='max',
                                      save_top_k=3)

# initialize datasets
image_train = ImageDataset(split=configuration['train_split_name'], dataset_type=configuration['dataset'], is_close=True,
                           closeness_factor=configuration['closeness_factor'])
image_val = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'], is_close=True,
                          closeness_factor=configuration['closeness_factor'])

# initialize lightning module
if configuration['pretraining']:
    c2ae = C2AELightning.load_from_checkpoint(configuration['pretraining_checkpoint'],
                                              n_classes=image_train.num_classes())
else:
    c2ae = C2AELightning(image_train.num_classes(), configuration['alpha'], learning_rate=configuration['lr'])

# initializing dataloaders
train_dataloader = DataLoader(image_train, batch_size=configuration['batch_size'], shuffle=True)
val_dataloader = DataLoader(image_val, batch_size=configuration['batch_size'], shuffle=False)

# training
trainer = pl.Trainer(max_epochs=configuration['max_epochs'], logger=wandb_logger, callbacks=[checkpoint_callback])
trainer.fit(model=c2ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# testing
#trainer.test(model=c2ae, dataloaders=test_dataloader)
