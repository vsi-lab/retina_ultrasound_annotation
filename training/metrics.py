# training/metrics.py
import torch
# import torch.nn.functional as F

@torch.no_grad()
def per_class_dice_from_logits(logits, target, num_classes, eps=1e-6, ignore_index=0):
    """
    Returns:
      overall: dict[class_id -> dice over all pixels]
      present_only: dict[class_id -> dice only for samples where class appears in GT]
      present_mask: dict[class_id -> bool whether class appears in batch GT]
    """
    pred = torch.argmax(logits, dim=1)  # [B,H,W]
    overall = {}
    present_only = {}
    present_mask = {}

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred == c)
        tgt_c  = (target == c)

        inter = (pred_c & tgt_c).sum().float()
        den   = pred_c.sum().float() + tgt_c.sum().float()
        dice  = (2 * inter + eps) / (den + eps)

        overall[c] = float(dice.cpu())

        is_present = tgt_c.any().item()
        present_mask[c] = bool(is_present)
        if is_present:
            present_only[c] = float(dice.cpu())
        else:
            present_only[c] = None

    return overall, present_only, present_mask


# @torch.no_grad()
# def dice_per_class_from_logits(logits, target, num_classes, ignore_index=0):
#     """
#     Backwards-compatible helper that returns a tensor [C] of overall dice,
#     including background at index 0.
#     """
#     overall, _, _ = per_class_dice_from_logits(logits, target, num_classes, ignore_index=ignore_index)
#     out = torch.zeros(num_classes, device=logits.device, dtype=torch.float32)
#     for c in range(num_classes):
#         if c == ignore_index:
#             out[c] = 0.0
#         else:
#             out[c] = overall.get(c, 0.0)
#     return out


@torch.no_grad()
def pixel_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean()


# def dice_loss_from_logits(logits, target, num_classes, eps=1e-6, ignore_index=0):
#     probs = torch.softmax(logits, dim=1)
#     oh = F.one_hot(target, num_classes).permute(0,3,1,2).float()
#     inter = (probs * oh).sum(dim=(2,3))
#     den   = (probs.pow(2) + oh.pow(2)).sum(dim=(2,3)) + eps
#     dice  = 2 * inter / den
#     keep = [c for c in range(num_classes) if c != ignore_index]
#     return 1.0 - dice[:, keep].mean()
#
#
# def focal_from_logits(logits, target, num_classes, gamma=2.0, ignore_index=0):
#     logp = F.log_softmax(logits, dim=1)
#     oh   = F.one_hot(target, num_classes).permute(0,3,1,2).float()
#     p    = torch.exp(logp)
#     loss = -((1.0 - p) ** gamma) * oh * logp
#     keep = [c for c in range(num_classes) if c != ignore_index]
#     return loss[:, keep, :, :].mean()
#
#
# def composite_loss(logits, target, num_classes,
#                    dice_w=0.7, focal_w=0.3, focal_gamma=2.0,
#                    ignore_index=0):
#     dl = dice_loss_from_logits(logits, target, num_classes, ignore_index=ignore_index)
#     fl = focal_from_logits(logits, target, num_classes, gamma=focal_gamma, ignore_index=ignore_index)
#     return dice_w * dl + focal_w * fl