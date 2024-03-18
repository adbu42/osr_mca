from contrastive_dataset import ContrastiveImageDataset
from torch.utils.data import DataLoader
from contrastive_architecture import SupConResNet
import torch


image_dataset = ContrastiveImageDataset(split='train', dataset_type='tiny')
train_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
images = torch.cat([train_features[0], train_features[1]], dim=0)
labels = torch.cat([train_labels[0], train_labels[1]], dim=0)
unet = SupConResNet()
output = unet.forward(images)
print('both:')
print(output.size())
print(labels.size())
