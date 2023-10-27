from datasets import load_dataset
from torch.utils.data import Dataset
from main_model.tiny_image_dataset import image_preprocessing
import numpy as np


class MNISTImageDataset(Dataset):
    def __init__(self, split='train', transform=None, target_transform=None):
        self.image_dataset = load_dataset('mnist', split=split)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img = self.image_dataset[idx]['image'].resize((64,64))
        img = image_preprocessing(img)
        label = self.image_dataset[idx]['label']

        # find a non match image in the dataset
        label_matched = True
        non_match_idx = 0
        while label_matched:
            non_match_idx = int(np.random.random_integers(0, len(self.image_dataset))-1)
            non_match_label = self.image_dataset[non_match_idx]['label']
            if non_match_label != label:
                label_matched = False
        non_match_img = self.image_dataset[non_match_idx]['image'].resize((64,64))
        non_match_img = image_preprocessing(non_match_img)

        # transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label, non_match_img, non_match_label
