from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import v2


class ContrastiveImageDataset(Dataset):
    def __init__(self, split='train', dataset_type='mnist', closeness_factor=1.0):
        self.image_key = 'image'
        self.augment_transform = None
        if dataset_type == 'mnist':
            image_dataset_huggingface = load_dataset('mnist', split=split)
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif dataset_type == 'tiny':
            image_dataset_huggingface = load_dataset('Maysee/tiny-imagenet', split=split)
            self.mean = [0.4802, 0.4481, 0.3975]
            self.std = [0.2770, 0.2691, 0.2821]
        elif dataset_type == 'svhn':
            image_dataset_huggingface = load_dataset('svhn', 'cropped_digits', split=split)
            self.mean = [0.4377, 0.4438, 0.4728]
            self.std = [0.1980, 0.2010, 0.1970]
        elif dataset_type == 'cifar':
            image_dataset_huggingface = load_dataset('cifar10', split=split)
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
            self.image_key = 'img'
        else:
            raise ValueError('Dataset not specified correctly!')


        train_transform = v2.Compose([
            v2.Resize((64, 64)),
            v2.RandomResizedCrop(size=64, scale=(0.2, 1.)),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([
                v2.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.ToTensor(),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

        self.transform = TwoCropTransform(train_transform)

        chosen_classes = len(image_dataset_huggingface.unique('label')) * closeness_factor
        self.image_dataset = image_dataset_huggingface.filter(lambda data_point: data_point['label'] <= chosen_classes)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img = self.image_dataset[idx][self.image_key]
        label = self.image_dataset[idx]['label']

        # transformations
        img = self.transform(img)
        labels = [label, label]
        return img, labels

    def num_classes(self):
        return len(self.image_dataset.unique(self.label_name))

    def reverse_normalization(self, image):
        inv_transform = v2.Normalize(-np.array(self.mean)/np.array(self.std),
                                     np.array([1.0, 1.0, 1.0])/self.std)
        return inv_transform(image)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
