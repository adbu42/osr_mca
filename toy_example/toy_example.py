from toy_modules import *


torch.set_float32_matmul_precision('high')

with open('../main_model/api_key.yml', 'r') as file:
    api_key = yaml.safe_load(file)

# initialize wandb
wandb.login(key=api_key['wandb_api_key'])
wandb_logger = WandbLogger(project='test_toy_example', save_dir='../main_model/tests/runs')


freeze_callback = FreezeEncoderOrDecoder()
toy_dataset_train, toy_dataset_valid = random_split(ToyDataset(), [18000, 2000])
toy_train_dataloader = DataLoader(toy_dataset_train, batch_size=128, shuffle=True)
toy_valid_dataloader = DataLoader(toy_dataset_valid, batch_size=128, shuffle=False)
toy_lightning = ToyLightning()
trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, callbacks=[freeze_callback])
trainer.fit(model=toy_lightning, train_dataloaders=toy_train_dataloader, val_dataloaders=toy_valid_dataloader)
