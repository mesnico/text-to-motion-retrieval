import torch
from torch import nn as nn

class Contrastive(nn.Module):
    def __init__(self, margin=0):
        super(Contrastive, self).__init__()
        self.margin = margin

    def compute_contrastive_loss(self, scores, max_violation=False):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class ContrastiveFixed(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation_after=0, **kwargs):
        super(ContrastiveFixed, self).__init__(margin=margin)
        self.max_violation_after = max_violation_after
        self.sim = lambda im, s: im.mm(s.t())

    def forward(self, comp_data, epoch, return_similarity_mat=False, **kwargs):
        im, s = comp_data['motion_emb'], comp_data['text_emb']
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        max_violation = epoch >= self.max_violation_after
        loss = self.compute_contrastive_loss(scores, max_violation)
        if return_similarity_mat:
            return loss, scores
        else:
            monitors = {}
            return loss, monitors