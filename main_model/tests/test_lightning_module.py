from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import yaml
import torch
import wandb

# optimize torch for cuda
torch.set_float32_matmul_precision('high')

# initialize configurations and wandb
with open('../config.yml', 'r') as file:
    configuration = yaml.safe_load(file)

with open('../api_key.yml', 'r') as file:
    api_key = yaml.safe_load(file)

wandb.login(key=api_key['wandb_api_key'])
wandb_logger = WandbLogger(project=configuration['project_name'], save_dir='runs')
wandb_logger.log_hyperparams(configuration)

image_train = ImageDataset(split=configuration['train_split_name'], dataset_type=configuration['dataset'], is_close=True,
                           closeness_factor=configuration['closeness_factor'])
image_test = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'], is_close=True,
                          closeness_factor=configuration['closeness_factor'])
c2ae = C2AELightning(image_train.num_classes(), configuration['alpha'], learning_rate=configuration['lr'])


# initializing dataloaders
train_dataloader = DataLoader(image_train, batch_size=configuration['batch_size'], shuffle=True)
test_dataloader = DataLoader(image_test, batch_size=configuration['batch_size'], shuffle=False)

# training
trainer = pl.Trainer(max_epochs=configuration['max_epochs'], logger=wandb_logger)
trainer.fit(model=c2ae, train_dataloaders=train_dataloader)

# testing
trainer.test(model=c2ae, dataloaders=test_dataloader)
