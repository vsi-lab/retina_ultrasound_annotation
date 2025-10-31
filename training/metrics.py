
import torch.nn.functional as F

def dice_per_class_from_logits(logits, target, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    oh = F.one_hot(target, num_classes).permute(0,3,1,2).float()
    inter = (probs * oh).sum(dim=(2,3))
    den = (probs.pow(2) + oh.pow(2)).sum(dim=(2,3)) + eps
    dice = 2*inter/den  # [B,C]
    return dice.mean(dim=0)  # [C]

def pixel_accuracy(logits, target):
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()
