import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class InfoNCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, comp_data, epoch, return_similarity_mat=False, **kwargs):
        im, s = comp_data['motion_emb'], comp_data['text_emb']

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * im @ s.t()
        logits_per_text = logits_per_image.t()

        # compute bidirectional CE loss
        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        if return_similarity_mat:
            return loss, logits_per_image
        else:
            monitors = {}
            return loss, monitors