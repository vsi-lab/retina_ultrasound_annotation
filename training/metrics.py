

import torch
import torch.nn.functional as F

@torch.no_grad()
def per_class_dice_from_logits(logits: torch.Tensor,
                               target: torch.Tensor,
                               num_classes: int,
                               ignore_index: int | None = 0,
                               eps: float = 1e-6):
    """
    Returns:
      overall: dict{cid -> dice over all samples (including zeros when absent)}
      present_only: dict{cid -> dice or None if class absent in that batch}
      counts: dict{cid -> number of present samples contributing}
    """
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)

    # one-hot target [B,C,H,W]
    target = target.clamp_min(0).clamp_max(num_classes - 1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    probs = F.softmax(logits, dim=1)

    # flatten safely
    probs_f  = probs.reshape(probs.size(0), probs.size(1), -1)
    target_f = target_1h.reshape(target_1h.size(0), target_1h.size(1), -1)

    overall = {}
    present_only = {}
    counts = {}

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            # still report 0 for convenience
            overall[c] = 0.0
            present_only[c] = None
            counts[c] = 0
            continue

        pc = probs_f[:, c, :]
        tc = target_f[:, c, :]

        intersect = (pc * tc).sum(dim=1)              # [B]
        denom     = pc.sum(dim=1) + tc.sum(dim=1)     # [B]
        dice_b    = (2 * intersect + eps) / (denom + eps)

        # mean over batch (including absent → tc.sum==0 implies tiny denom; but we want zeros in "overall")
        overall[c] = float(dice_b.mean().item())

        # present-only mean
        present_mask = (tc.sum(dim=1) > 0)
        if present_mask.any():
            present_only[c] = float(dice_b[present_mask].mean().item())
            counts[c] = int(present_mask.sum().item())
        else:
            present_only[c] = None
            counts[c] = 0

    return overall, present_only, counts


@torch.no_grad()
def pixel_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean()
