from main_model.lightning_module import C2AELightning
from main_model.dataset import ImageDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import torch
import torch.nn.functional as F
import numpy as np


def osr(model, configuration):
    # instantiate dataset
    image_dataset_closed = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                                        chosen_classes=configuration['chosen_classes'])
    image_dataset_open = ImageDataset(split=configuration['test_split_name'], dataset_type=configuration['dataset'],
                                      chosen_classes=configuration['chosen_classes'], is_close=False)
    closed_num_classes = image_dataset_closed.num_classes()
    closed_dataloader = DataLoader(image_dataset_closed, batch_size=configuration['batch_size'], shuffle=True)
    open_dataloader = DataLoader(image_dataset_open, batch_size=configuration['batch_size'], shuffle=True)
    model.eval()

    l1_loss = nn.L1Loss()
    label_possibilities = [torch.full((1, configuration['batch_size']), k).squeeze() for k in range(closed_num_classes)]

    closed_error_list = []
    closed_counter = 0
    for features, _, _, _ in closed_dataloader:
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
            open_error_list.append(minimum_reconstruction_error)
        open_counter += 1
        print(open_counter / len(open_dataloader))

    np_open_error_list = np.array(closed_error_list)
    np_closed_error_list = np.array(open_error_list)
    return np_open_error_list, np_closed_error_list


def main():
    with open('configs/config.yml', 'r') as file:
        configs = yaml.safe_load(file)

    # instantiate model and loss
    model_out_of_scope = C2AELightning.load_from_checkpoint(configs['checkpoint'],
                                                            n_classes=len(configs['chosen_classes']),
                                                            architecture=configs['architecture'])

    np_open_error_list_out, np_closed_error_list_out = osr(model_out_of_scope, configs)

    np.savetxt("tests/match_errors.csv", np_open_error_list_out, delimiter=",")
    np.savetxt("tests/non_match_errors.csv", np_closed_error_list_out, delimiter=",")


if __name__ == "__main__":
    main()
