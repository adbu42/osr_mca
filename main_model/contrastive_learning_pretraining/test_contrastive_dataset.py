from contrastive_dataset import ContrastiveImageDataset
from torch.utils.data import DataLoader
from main_model.tests.testing_helper_functions import show_image
import yaml
import torch


with open('../configs/config.yml', 'r') as file:
    configuration = yaml.safe_load(file)

image_dataset = ContrastiveImageDataset(split=configuration['train_split_name'],
                             dataset_type=configuration['dataset'], closeness_factor=configuration['closeness_factor'])
train_dataloader = DataLoader(image_dataset, batch_size=3, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
labels = torch.cat([train_labels[0], train_labels[1]], dim=0)
print(f"labels: {labels[0]}, {labels[1]}, {labels[2]}, {labels[3]}, {labels[4]}, {labels[5]}")

images = torch.cat([train_features[0], train_features[1]], dim=0)
show_image(images[0], image_dataset)
show_image(images[1], image_dataset)
show_image(images[2], image_dataset)
show_image(images[3], image_dataset)
show_image(images[4], image_dataset)
show_image(images[5], image_dataset)
