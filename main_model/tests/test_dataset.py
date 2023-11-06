from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
from testing_helper_functions import show_image
import yaml


with open('../config.yml', 'r') as file:
    configuration = yaml.safe_load(file)

image_dataset = ImageDataset(split=configuration['train_split_name'],
                             dataset_type=configuration['dataset'],
                             is_close=True, closeness_factor=configuration['closeness_factor'])
train_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True)
train_features, train_labels, train_non_match_features, train_non_match_labels = next(iter(train_dataloader))
print(f"labels: {train_labels}")
print(f"non_match_labels: {train_non_match_labels}")
show_image(train_features[0])
show_image(train_features[1])
show_image(train_non_match_features[0])
show_image(train_non_match_features[1])
