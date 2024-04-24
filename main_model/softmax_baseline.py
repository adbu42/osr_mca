from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


with open('configs/config.yml', 'r') as file:
    configuration = yaml.safe_load(file)

batch_size = configuration['batch_size']
# instantiate model and loss
model = C2AELightning.load_from_checkpoint(configuration['checkpoint'], n_classes=len(configuration['chosen_classes']),
                                           architecture=configuration['architecture'])

image_dataset_closed = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                                    chosen_classes=configuration['chosen_classes'])
image_dataset_open = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                                  chosen_classes=configuration['chosen_classes'], is_close=False)
closed_num_classes = image_dataset_closed.num_classes()
closed_dataloader = DataLoader(image_dataset_closed, batch_size=batch_size, shuffle=True)
open_dataloader = DataLoader(image_dataset_open, batch_size=batch_size, shuffle=True)
model.eval()

# closed set predictions
fake_condition_vector = F.one_hot(torch.full((1, batch_size), 1).squeeze(),
                                  closed_num_classes).float()
all_closed_set_predictions = torch.zeros((len(image_dataset_closed)))
i = 0
for features, _, _, _ in closed_dataloader:
    if len(features) != batch_size:
        fake_condition_vector = F.one_hot(torch.full((1, len(features)), 1).squeeze(),
                                          closed_num_classes).float()
    prediction = model(features.cuda().detach(), fake_condition_vector.cuda().detach())[1].cpu().detach()
    all_closed_set_predictions[i*batch_size:i*batch_size+batch_size] = torch.max(prediction, dim=1)[0]
    i += 1

# open set predictions
fake_condition_vector = F.one_hot(torch.full((1, batch_size), 1).squeeze(),
                                  closed_num_classes).float()
all_open_set_predictions = torch.zeros((len(image_dataset_open)))
i = 0
for features, _, _, _ in open_dataloader:
    if len(features) != batch_size:
        fake_condition_vector = F.one_hot(torch.full((1, len(features)), 1).squeeze(),
                                          closed_num_classes).float()
    prediction = model(features.cuda().detach(), fake_condition_vector.cuda().detach())[1].cpu().detach()
    all_open_set_predictions[i*batch_size:i*batch_size+batch_size] = torch.max(prediction, dim=1)[0]
    i += 1

# calculate auc for baseline
combined_errors = np.concatenate([all_open_set_predictions, all_closed_set_predictions])

# Create labels (0 for error_dist1, 1 for error_dist2)
labels = np.concatenate([np.zeros(len(all_open_set_predictions)), np.ones(len(all_closed_set_predictions))])

auc_score = roc_auc_score(labels, combined_errors)

print(f"AUC score for the combined error distributions: {auc_score:.4f}")
