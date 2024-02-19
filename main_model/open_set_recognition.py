from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import torch
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
import numpy as np

cutoff_point = torch.tensor(0.1)

with open('configs/config.yml', 'r') as file:
    configuration = yaml.safe_load(file)


# instantiate dataset
image_dataset_closed = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                             closeness_factor=configuration['closeness_factor'])
image_dataset_open = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                             closeness_factor=configuration['closeness_factor'], is_close=False)


closed_num_classes = image_dataset_closed.num_classes()
closed_dataloader = DataLoader(image_dataset_closed, batch_size=configuration['batch_size'], shuffle=True)
open_dataloader = DataLoader(image_dataset_open, batch_size=configuration['batch_size'], shuffle=True)

# instantiate model and loss
model = C2AELightning.load_from_checkpoint(configuration['checkpoint'], n_classes=closed_num_classes,
                                           architecture=configuration['architecture'])
model.eval()

accuracy_metric = MulticlassAccuracy(num_classes=closed_num_classes)
l1_loss = nn.L1Loss()

accuracy = []
label_possibilities = [torch.full((1, configuration['batch_size']), k).squeeze() for k in range(closed_num_classes)]
is_classified_closed = []
is_classified_open = []

closed_error_list = []
closed_counter = 0
for features, labels, _, _ in closed_dataloader:
    conditional_vector = F.one_hot(labels, closed_num_classes).float()
    classification = model(features.cuda().detach(), conditional_vector.cuda().detach())[1].cpu().detach()
    accuracy.append(accuracy_metric(classification, labels.detach()).cpu().detach())

    if len(features) != configuration['batch_size']:
        label_possibilities = [torch.full((1, len(features)), k).squeeze()
                               for k in range(closed_num_classes)]
    reconstruction_error_list = []
    for label_possibility in label_possibilities:
        conditional_vector = F.one_hot(label_possibility, closed_num_classes).float()
        conditional_vector[conditional_vector == 0] = -1
        reconstruction = model(features.cuda().detach(), conditional_vector.cuda().detach())[0].cpu().detach()
        reconstruction_error_batch = []
        for i in range(len(reconstruction)):
            reconstruction_error = l1_loss(reconstruction[i], features[i].detach())
            reconstruction_error_batch.append(reconstruction_error.cpu().detach())
        reconstruction_error_list.append(reconstruction_error_batch)

    for item in range(len(reconstruction_error_list[0])):
        minimum_reconstruction_error = 10
        for reconstruction_error in reconstruction_error_list:
            if reconstruction_error[item] < minimum_reconstruction_error:
                minimum_reconstruction_error = reconstruction_error[item]
        if minimum_reconstruction_error <= cutoff_point:
            is_classified_closed.append(1)
        else:
            is_classified_closed.append(0)
        closed_error_list.append(minimum_reconstruction_error)
    closed_counter += 1
    print(closed_counter/len(closed_dataloader))

label_possibilities = [torch.full((1, configuration['batch_size']), k).squeeze() for k in range(closed_num_classes)]

open_counter = 0
open_error_list = []
for features, _, _, _ in open_dataloader:
    if len(features) != configuration['batch_size']:
        label_possibilities = [torch.full((1, len(features)), k).squeeze()
                               for k in range(closed_num_classes)]
    reconstruction_error_list = []
    for label_possibility in label_possibilities:
        conditional_vector = F.one_hot(label_possibility, closed_num_classes).float()
        conditional_vector[conditional_vector == 0] = -1
        reconstruction = model(features.cuda().detach(), conditional_vector.cuda().detach())[0].cpu().detach()
        reconstruction_error_batch = []
        for i in range(len(reconstruction)):
            reconstruction_error = l1_loss(reconstruction[i], features[i].detach())
            reconstruction_error_batch.append(reconstruction_error.cpu().detach())
        reconstruction_error_list.append(reconstruction_error_batch)

    for item in range(len(reconstruction_error_list[0])):
        minimum_reconstruction_error = 10
        for reconstruction_error in reconstruction_error_list:
            if reconstruction_error[item] < minimum_reconstruction_error:
                minimum_reconstruction_error = reconstruction_error[item]
        if minimum_reconstruction_error >= cutoff_point:
            is_classified_open.append(1)
        else:
            is_classified_open.append(0)
        open_error_list.append(minimum_reconstruction_error) #here
    open_counter += 1
    print(open_counter / len(open_dataloader))

print(f'accuracy: {np.mean(accuracy)}')
print(f'false closed: {is_classified_closed.count(0)/len(is_classified_closed)}')
print(f'true closed: {is_classified_closed.count(1)/len(is_classified_closed)}')
print(f'false open: {is_classified_open.count(0)/len(is_classified_open)}')
print(f'true open: {is_classified_open.count(1)/len(is_classified_open)}')

np_open_error_list = np.array(closed_error_list)
np_closed_error_list = np.array(open_error_list)
np.savetxt("tests/match_errors.csv", np_open_error_list, delimiter=",")
np.savetxt("tests/non_match_errors.csv", np_closed_error_list, delimiter=",")
