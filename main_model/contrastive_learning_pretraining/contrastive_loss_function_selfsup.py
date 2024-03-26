import torch
import torch.nn as nn
import torch.nn.functional as F


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max
    return logits


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None
        self.p = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1, p=2)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if labels.size(0) != self.last_local_batch_size:
            mask = torch.zeros((labels.size(0), labels.size(0))).to(device)
            for col in range(mask.size(0)):
                position = (col + mask.size(0) // 2) % mask.size(0)
                mask[col, position] = 1
            self.last_local_batch_size = labels.size(0)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask), 1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device), 0
            )
            self.p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # compute logits
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        loss = compute_cross_entropy(self.p, logits)

        return loss
