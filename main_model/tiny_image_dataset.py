from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class TinyImageDataset(Dataset):
    def __init__(self, split='train', transform=None, target_transform=None):
        self.image_dataset = load_dataset('Maysee/tiny-imagenet', split=split)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img = self.image_dataset[idx]['image']
        img = pil_to_tensor(img)
        img = img.float()/255  # set the values between 0 and 1
        if img.size(0) == 1:
            img = img.expand(3, 64, 64)
        label = self.image_dataset[idx]['label']
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
