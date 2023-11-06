import torch

from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
from main_model.u_net_architecture import UNet

image_dataset = ImageDataset(split='train', dataset_type='tiny')
train_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True)
train_features, train_labels, _, _ = next(iter(train_dataloader))

unet = UNet(3, 3, 5)
output = unet.forward(train_features, torch.Tensor([[-1, 1, -1, -1, -1], [-1, -1, -1, 1, -1]]))
print(output.size())
