import torch
from contrastive_loss_function import MultiPosConLoss

labels = torch.tensor([1, 2, 1, 5]).to('cuda:0')
features = torch.tensor([[0.2, 0.4, 0.3, 0.7], [0.2, 0.1, 0.8, 0.4], [0.2, 0.4, 0.3, 0.7], [0.6, 0.2, 0.3, 0.1]]).to('cuda:0')
loss_fn = MultiPosConLoss()
loss = loss_fn(features, labels)
print(loss)
