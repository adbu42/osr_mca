from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.functional import pil_to_tensor


def image_preprocessing(pil_image):
    img = pil_to_tensor(pil_image)
    img = img.float() / 255  # set the values between 0 and 1
    if img.size(0) == 1:
        img = img.expand(3, 64, 64)
    return img


class ImageDataset(Dataset):
    def __init__(self, split='train', dataset_type='mnist', closeness_factor=1.0, is_close=True, transform=None,
                 target_transform=None):
        self.image_key = 'image'
        if dataset_type == 'mnist':
            image_dataset_huggingface = load_dataset('mnist', split=split)
        elif dataset_type == 'tiny':
            image_dataset_huggingface = load_dataset('Maysee/tiny-imagenet', split=split)
        elif dataset_type == 'svhn':
            image_dataset_huggingface = load_dataset('svhn', 'cropped_digits', split=split)
        elif dataset_type == 'cifar':
            image_dataset_huggingface = load_dataset('cifar10', split=split)
            self.image_key = 'img'

        self.transform = transform
        self.target_transform = target_transform
        chosen_classes = len(image_dataset_huggingface.unique('label')) * closeness_factor
        if is_close:
            self.image_dataset = image_dataset_huggingface.filter(
                lambda data_point: data_point['label'] <= chosen_classes)
        else:
            self.image_dataset = image_dataset_huggingface.filter(
                lambda data_point: data_point['label'] > chosen_classes)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img = self.image_dataset[idx][self.image_key].resize((64, 64))
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
        non_match_img = self.image_dataset[non_match_idx][self.image_key].resize((64, 64))
        non_match_img = image_preprocessing(non_match_img)

        # transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label, non_match_img, non_match_label

    def num_classes(self):
        return len(self.image_dataset.unique('label'))
