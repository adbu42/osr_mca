from main_model.tiny_image_dataset import TinyImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

tiny_image = TinyImageDataset(split='train')
train_dataloader = DataLoader(tiny_image, batch_size=2, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
plt.figure()
plt.imshow(train_features[0].permute(1, 2, 0))
plt.show()
plt.figure()
plt.imshow(train_features[1].permute(1, 2, 0))
plt.show()
