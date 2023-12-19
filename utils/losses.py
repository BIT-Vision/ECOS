import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input_mask, cls_gt):
    cls_gt = torch.argmax(cls_gt, dim=1)
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:, i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt == (i + 1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        target = torch.argmax(target, dim=1)
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class Loss(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000):
        super().__init__()
        self.bce = BootstrappedCE(start_warm, end_warm)

    def forward(self, logits, probs, masks, n_objs, step=0):
        b, t = logits.shape[:2]

        total_loss = 0
        ce_loss = []
        dice = []
        for ti in range(t):
            for bi in range(b):
                loss, p = self.bce(logits[bi, ti:ti + 1, :n_objs[bi]], masks[bi:bi + 1, ti], step)
                ce_loss.append(loss)

            d_loss = dice_loss(probs[:, ti, 1:], masks[:, ti])
            dice.append(d_loss)

        ce_loss = torch.stack(ce_loss, dim=0).mean()
        dice = torch.stack(dice, dim=0).mean()

        loss = ce_loss + dice
        loss_stats = {'cls_loss': ce_loss.item(), 'iou_loss': dice.item()}
        return loss, loss_stats
