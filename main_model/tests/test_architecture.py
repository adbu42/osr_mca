from main_model.tiny_image_dataset import TinyImageDataset
from torch.utils.data import DataLoader
from main_model.u_net_architecture import UNet

tiny_image = TinyImageDataset(split='train')
train_dataloader = DataLoader(tiny_image, batch_size=2, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))

unet = UNet(3, 3)
output = unet.forward(train_features)
print(output.size())
