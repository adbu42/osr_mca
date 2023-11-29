from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import yaml


with open('../config.yml', 'r') as file:
    configuration = yaml.safe_load(file)


# instantiate dataset
mnist_image = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                           closeness_factor=configuration['closeness_factor'])
num_classes = mnist_image.num_classes()
test_dataloader = DataLoader(mnist_image, batch_size=configuration['batch_size'], shuffle=True)

# instantiate model and loss
model = C2AELightning.load_from_checkpoint(configuration['checkpoint'], n_classes=num_classes)
model.eval()
loss = nn.L1Loss()

match_errors = []
non_match_errors = []

# evaluation
for test_features, test_labels, non_match_features, non_match_labels in test_dataloader:
    # one-hot encode the features
    conditional_vector = F.one_hot(test_labels, num_classes).float()
    non_match_conditional_vector = F.one_hot(non_match_labels, num_classes).float()

    conditional_vector[conditional_vector == 0] = -1
    non_match_conditional_vector[non_match_conditional_vector == 0] = -1
    # predict with the model
    # detach all tensor so no gradients are computed and the tensors are not kept in memory
    match_prediction, _ = model(test_features.cuda().detach(), conditional_vector.cuda().detach())
    non_match_prediction, _ = model(test_features.cuda().detach(), non_match_conditional_vector.cuda().detach())
    for i in range(len(match_prediction)):
        match_errors.append(loss(match_prediction[i].detach(), test_features[i].cuda().detach()).cpu())
        non_match_errors.append(loss(non_match_prediction[i].detach(), test_features[i].cuda().detach()).cpu())
    print(len(match_errors)/len(mnist_image))

np_match_errors = np.array(match_errors)
np_non_match_errors = np.array(non_match_errors)
np.savetxt("match_errors.csv", np_match_errors, delimiter=",")
np.savetxt("non_match_errors.csv", np_non_match_errors, delimiter=",")
