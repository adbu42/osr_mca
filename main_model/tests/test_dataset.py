from main_model.tiny_image_dataset import TinyImageDataset
from main_model.mnist_dataset import MNISTImageDataset
from torch.utils.data import DataLoader
from testing_helper_functions import show_image

mnist_image = MNISTImageDataset(split='train')
train_dataloader = DataLoader(mnist_image, batch_size=2, shuffle=True)
train_features, train_labels, train_non_match_features, train_non_match_labels = next(iter(train_dataloader))
print(f"labels: {train_labels}")
print(f"non_match_labels: {train_non_match_labels}")
show_image(train_features[0])
show_image(train_features[1])
show_image(train_non_match_features[0])
show_image(train_non_match_features[1])
