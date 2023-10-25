from main_model.lightning_module import C2AELightning
from main_model.tiny_image_dataset import TinyImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
tiny_image = TinyImageDataset(split='train')
train_dataloader = DataLoader(tiny_image, batch_size=2, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
plt.figure()
plt.imshow(train_features[0].permute(1, 2, 0))
plt.show()
plt.figure()
plt.imshow(train_features[1].permute(1, 2, 0))
plt.show()

# evaluation
model = C2AELightning.load_from_checkpoint("test_u_net/dfoqpx0v/checkpoints/epoch=0-step=200.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(train_features.cuda())
plt.figure()
plt.imshow(y_hat[0].permute(1, 2, 0).cpu().detach())
plt.show()
plt.figure()
plt.imshow(y_hat[1].permute(1, 2, 0).cpu().detach())
plt.show()
