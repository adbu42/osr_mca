from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
from testing_helper_functions import show_image
import torch.nn.functional as F
import yaml
from torch.nn import Softmax
import torch


softmax = Softmax(dim=0)
# load configuration
with open('../configs/config.yml', 'r') as file:
    configuration = yaml.safe_load(file)

image_dataset = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                             is_close=True, closeness_factor=configuration['closeness_factor'])
num_classes = 7
test_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True)
test_features, _, _, _ = next(iter(test_dataloader))
show_image(test_features[0])
show_image(test_features[1])

# evaluation
model = C2AELightning.load_from_checkpoint(configuration['checkpoint'], n_classes=num_classes,
                                           architecture=configuration['architecture'])

# disable randomness, dropout, etc...
model.eval()

label_possibilities = [torch.full((1, 2), k).squeeze() for k in range(num_classes)]

for labels in label_possibilities:
    conditional_vector = F.one_hot(labels, num_classes).float()
    conditional_vector[conditional_vector == 0] = -1
    good_prediction, _ = model(test_features.cuda(), conditional_vector.cuda())
    show_image(good_prediction[0].cpu().detach())
    show_image(good_prediction[1].cpu().detach())
