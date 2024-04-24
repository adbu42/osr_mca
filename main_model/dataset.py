from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms import AutoAugmentPolicy


class ImageDataset(Dataset):
    def __init__(self, split='train', dataset_type='mnist', chosen_classes=None, is_close=True, augmented=False):
        self.image_key = 'image'
        self.augment_transform = None
        if dataset_type == 'mnist':
            image_dataset_huggingface = load_dataset('mnist', split=split)
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.augment_transform = v2.AutoAugment(policy=AutoAugmentPolicy("svhn"))
        elif dataset_type == 'tiny':
            image_dataset_huggingface = load_dataset('Maysee/tiny-imagenet', split=split)
            self.mean = [0.4802, 0.4481, 0.3975]
            self.std = [0.2770, 0.2691, 0.2821]
            self.augment_transform = v2.AutoAugment(policy=AutoAugmentPolicy("imagenet"))
        elif dataset_type == 'svhn':
            image_dataset_huggingface = load_dataset('svhn', 'cropped_digits', split=split)
            self.mean = [0.4377, 0.4438, 0.4728]
            self.std = [0.1980, 0.2010, 0.1970]
            self.augment_transform = v2.AutoAugment(policy=AutoAugmentPolicy("svhn"))
        elif dataset_type == 'cifar':
            image_dataset_huggingface = load_dataset('cifar10', split=split)
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
            self.image_key = 'img'
            self.augment_transform = v2.AutoAugment(policy=AutoAugmentPolicy("cifar10"))
        elif dataset_type == 'cifar_more':
            image_dataset_huggingface_more = load_dataset('cifar100', split=split)
            image_dataset_huggingface = load_dataset('cifar10', split=split)
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
            self.image_key = 'img'
            self.augment_transform = v2.AutoAugment(policy=AutoAugmentPolicy("cifar10"))
        else:
            raise ValueError('Dataset not specified correctly!')

        if split == "train" and augmented:
            self.transform = v2.Compose([
                v2.Resize((32, 32)),
                v2.ToTensor(),
                self.augment_transform,
                v2.Normalize(self.mean, self.std)
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize((32, 32)),
                v2.ToTensor(),
                v2.Normalize(self.mean, self.std)
            ])

        if is_close:
            self.image_dataset = image_dataset_huggingface.filter(
                lambda data_point: data_point['label'] in chosen_classes)
            self.label_name = 'label'
        elif dataset_type == 'cifar_more' and not is_close:
            self.image_dataset = image_dataset_huggingface_more.filter(
                lambda data_point: data_point['fine_label'] <= 40)
            self.label_name = 'fine_label'
        else:
            self.image_dataset = image_dataset_huggingface.filter(
                lambda data_point: data_point['label'] not in chosen_classes)
            self.label_name = 'label'

        if chosen_classes == [3, 4, 5, 6, 7, 8, 9]:
            self.mapping_vector = [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        elif chosen_classes == [0, 1, 2, 6, 7, 8, 9]:
            self.mapping_vector = [0, 1, 2, 7, 8, 9, 3, 4, 5, 6]
        elif dataset_type != 'tiny':
            self.mapping_vector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            self.mapping_vector = [x for x in range(200)]

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img = self.image_dataset[idx][self.image_key]
        label = self.image_dataset[idx][self.label_name]

        # find a non match image in the dataset
        label_matched = True
        non_match_idx = 0
        while label_matched:
            non_match_idx = int(np.random.random_integers(0, len(self.image_dataset)) - 1)
            non_match_label = self.image_dataset[non_match_idx][self.label_name]
            if non_match_label != label:
                label_matched = False
        non_match_img = self.image_dataset[non_match_idx][self.image_key]

        # transformations
        img = self.transform(img)
        non_match_img = self.transform(non_match_img)
        return img, self.mapping_vector[label], non_match_img, self.mapping_vector[non_match_label]

    def num_classes(self):
        return len(self.image_dataset.unique(self.label_name))

    def reverse_normalization(self, image):
        inv_transform = v2.Normalize(-np.array(self.mean)/np.array(self.std),
                                     np.array([1.0, 1.0, 1.0])/self.std)
        return inv_transform(image)
