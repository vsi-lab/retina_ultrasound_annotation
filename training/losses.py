import torch
import torch.nn.functional as F

def _flatten_logits_targets(logits: torch.Tensor,
                            target: torch.Tensor,
                            ignore_index: int | None = None):
    """
    logits: [B,C,H,W]
    target: [B,H,W]
    returns logits_2d [N,C], target_1d [N]
    """
    if ignore_index is not None:
        mask = (target != ignore_index)
    else:
        mask = torch.ones_like(target, dtype=torch.bool)

    # [B,C,H,W] -> [B,H,W,C] -> boolean index -> [N,C]
    logits_2d = logits.permute(0, 2, 3, 1)[mask].reshape(-1, logits.size(1))
    target_1d = target[mask].reshape(-1)
    return logits_2d, target_1d


def dice_loss(logits: torch.Tensor,
              target: torch.Tensor,
              num_classes: int,
              ignore_index: int = 0,
              eps: float = 1e-6) -> torch.Tensor:
    """
    Multiclass soft Dice loss; ignores 'ignore_index' (usually background=0).
    """
    # one-hot target, shape [B,C,H,W]
    target = target.clamp_min(0).clamp_max(num_classes - 1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    probs = F.softmax(logits, dim=1)

    # reshape safely (NO .view)
    probs_f  = probs.reshape(probs.size(0), probs.size(1), -1)
    target_f = target_1h.reshape(target_1h.size(0), target_1h.size(1), -1)

    # ignore background
    cids = [c for c in range(num_classes) if c != ignore_index]
    probs_f  = probs_f[:, cids, :]
    target_f = target_f[:, cids, :]

    # per-class dice
    intersect = (probs_f * target_f).sum(dim=2)
    denom     = probs_f.sum(dim=2) + target_f.sum(dim=2)
    dice_c    = (2 * intersect + eps) / (denom + eps)

    loss = 1.0 - dice_c.mean()
    return loss


def focal_loss(logits: torch.Tensor,
               target: torch.Tensor,
               gamma: float = 2.0,
               ignore_index: int | None = 0) -> torch.Tensor:
    """
    Standard multiclass focal loss on logits; ignores 'ignore_index'.
    """
    logits_2d, target_1d = _flatten_logits_targets(logits, target, ignore_index)
    logpt = F.log_softmax(logits_2d, dim=1)
    pt    = logpt.exp()
    # gather the prob of the true class
    logpt_t = logpt.gather(1, target_1d.unsqueeze(1)).squeeze(1)
    pt_t    = pt.gather(1, target_1d.unsqueeze(1)).squeeze(1)
    loss    = -((1 - pt_t) ** gamma) * logpt_t
    return loss.mean()


def composite_loss(logits: torch.Tensor,
                   target: torch.Tensor,
                   num_classes: int,
                   dice_w: float = 1.0,
                   focal_w: float = 0.0,
                   focal_gamma: float = 2.0,
                   ignore_index: int = 0) -> torch.Tensor:
    # ---- add this block ----
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
    # ------------------------

    ld = dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    lf = torch.tensor(0.0, device=logits.device)
    if focal_w > 0:
        lf = focal_loss(logits, target, gamma=focal_gamma, ignore_index=ignore_index)
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    return dice_w * ld + focal_w * lf + 0.1 * ce  # e.g., ce weight 0.2

# # training/losses.py
# import torch
# import torch.nn.functional as F
#
# def dice_loss_from_logits(logits, target, num_classes, eps=1e-6):
#     """
#     Compute soft Dice loss directly from unnormalized logits.
#
#     This loss measures the spatial overlap between predicted segmentation maps
#     and ground-truth masks, encouraging class-wise region consistency. It is
#     differentiable and particularly suited for medical image segmentation tasks
#     such as retinal detachment delineation.
#
#     Args:
#         logits (Tensor): Raw model outputs of shape (B, C, H, W).
#         target (Tensor): Ground-truth integer mask of shape (B, H, W).
#         num_classes (int): Number of classes including background.
#         eps (float, optional): Small epsilon to avoid division by zero.
#
#     Returns:
#         Tensor: Scalar Dice loss averaged over foreground classes (C-1).
#
#     References:
#         - Milletari et al., *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*, 3DV 2016.
#         - Sudre et al., *Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations*, DLMIA 2017.
#     """
#     probs = torch.softmax(logits, dim=1)
#     oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
#     inter = (probs * oh).sum(dim=(2, 3))
#     den = (probs.pow(2) + oh.pow(2)).sum(dim=(2, 3)) + eps
#     dice = 2 * inter / den
#     return 1 - dice[:, 1:].mean()
#
#
# def focal_from_logits(logits, target, num_classes, gamma=2.0):
#     """
#     Compute Focal loss from unnormalized logits.
#
#     The focal term down-weights easy examples and focuses training on harder
#     misclassified pixels, which helps when dealing with severe class imbalance
#     (e.g., small retinal detachment regions vs. large background).
#
#     Args:
#         logits (Tensor): Raw model outputs of shape (B, C, H, W).
#         target (Tensor): Ground-truth integer mask of shape (B, H, W).
#         num_classes (int): Number of segmentation classes.
#         gamma (float, optional): Focusing parameter (>0). Larger values increase
#             the down-weighting of easy examples. Default is 2.0.
#
#     Returns:
#         Tensor: Scalar focal loss averaged over foreground classes.
#
#     References:
#         - Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
#     """
#     logp = F.log_softmax(logits, dim=1)
#     oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
#     p = torch.exp(logp)
#     loss = -((1 - p) ** gamma) * oh * logp
#     return loss[:, 1:, :, :].mean()
#
#
# def composite_loss(logits, target, num_classes, dice_w=0.7, focal_w=0.3, focal_gamma=2.0):
#     """
#     Weighted composite loss combining Dice and Focal components.
#
#     This hybrid objective stabilizes early training by leveraging the global
#     overlap sensitivity of Dice loss and the local pixel-wise reweighting of
#     Focal loss. It has been empirically shown to improve boundary precision
#     and convergence stability in medical segmentation networks like TransUNet.
#
#     Args:
#         logits (Tensor): Model output logits (B, C, H, W).
#         target (Tensor): Ground-truth mask (B, H, W).
#         num_classes (int): Total number of segmentation classes.
#         dice_w (float, optional): Weight for Dice loss component. Default 0.7.
#         focal_w (float, optional): Weight for Focal loss component. Default 0.3.
#         focal_gamma (float, optional): Focal focusing parameter. Default 2.0.
#
#     Returns:
#         Tensor: Weighted sum of Dice and Focal losses.
#
#     Example:
#         >>> loss = composite_loss(pred_logits, gt_mask, num_classes=3)
#         >>> loss.backward()
#
#     Notes:
#         For most retinal detachment segmentation runs:
#         - Dice emphasizes correct regional coverage.
#         - Focal emphasizes detection of thin membrane boundaries.
#     """
#     dl = dice_loss_from_logits(logits, target, num_classes)
#     fl = focal_from_logits(logits, target, num_classes, gamma=focal_gamma)
#     return dice_w * dl + focal_w * fl