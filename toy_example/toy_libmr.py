from toy_modules import *

# instantiate dataset
toy_dataset_train, toy_dataset_valid = random_split(ToyDataset(), [18000, 2000])

toy_dataloader = DataLoader(toy_dataset_valid, batch_size=128, shuffle=True)
num_classes = 4
# instantiate model and loss
model = ToyLightning.load_from_checkpoint('tests/runs/test_toy_example/l5hhgi6q/checkpoints/epoch=299-step=42300.ckpt')
model.eval()
loss = nn.L1Loss()

match_errors = []
non_match_errors = []

# evaluation
for test_features, test_labels, non_match_features, non_match_labels in toy_dataloader:
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
    print(len(match_errors)/len(toy_dataset_valid))

np_match_errors = np.array(match_errors)
np_non_match_errors = np.array(non_match_errors)
np.savetxt("../main_model/tests/match_errors.csv", np_match_errors, delimiter=",")
np.savetxt("../main_model/tests/non_match_errors.csv", np_non_match_errors, delimiter=",")
