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

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1, p=2)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        mask = torch.eq(labels.view(-1, 1),
                        labels.contiguous().view(1, -1)).float().to(device)
        self.logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device), 0
        )

        mask = mask * self.logits_mask

        # compute logits
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss
