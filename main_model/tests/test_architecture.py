from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
from main_model.architectures.wide_net import *

image_dataset = ImageDataset(split='train', dataset_type='tiny')
train_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True)
train_features, train_labels, _, _ = next(iter(train_dataloader))

unet = WideResNet(5)
output, classification = unet.forward(train_features, torch.Tensor([[-1, 1, -1, -1, -1], [-1, -1, -1, 1, -1]]))
print('both:')
print(output.size())
print(classification.size())
