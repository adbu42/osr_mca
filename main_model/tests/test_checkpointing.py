from main_model.lightning_module import C2AELightning
from main_model.tiny_image_dataset import TinyImageDataset
from main_model.mnist_dataset import MNISTImageDataset
from torch.utils.data import DataLoader
from testing_helper_functions import show_image
import torch.nn.functional as F

num_classes = 10
mnist_image = MNISTImageDataset(split='test')
test_dataloader = DataLoader(mnist_image, batch_size=2, shuffle=True)
test_features, test_labels, non_match_features, non_match_labels = next(iter(test_dataloader))
show_image(test_features[0])
show_image(test_features[1])
print(f"label of first image: {test_labels[0]}")
print(f"label of second image: {test_labels[1]}")
print(f"label of first non_match image: {non_match_labels[0]}")
print(f"label of second non_match image: {non_match_labels[1]}")

# evaluation
model = C2AELightning.load_from_checkpoint("runs/test_u_net_conditional_mnist/vuqvytbv/checkpoints/epoch=2-step=9000.ckpt", n_classes=num_classes)

# disable randomness, dropout, etc...
model.eval()

# one-hot encode the features
conditional_vector = F.one_hot(test_labels, num_classes).float()
non_match_conditional_vector = F.one_hot(non_match_labels, num_classes).float()

conditional_vector[conditional_vector == 0] = -1
non_match_conditional_vector[non_match_conditional_vector == 0] = -1


# predict with the model
good_prediction = model(test_features.cuda(), conditional_vector.cuda())
bad_prediction = model(test_features.cuda(), non_match_conditional_vector.cuda())
show_image(good_prediction[0].cpu().detach())
show_image(good_prediction[1].cpu().detach())
show_image(bad_prediction[0].cpu().detach())
show_image(bad_prediction[1].cpu().detach())
