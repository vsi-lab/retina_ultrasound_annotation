# eval_from_masks.py
import glob, os, cv2, numpy as np, torch
from training.metrics import per_class_dice_from_logits

NUM_CLASSES = 5          # 0=bg, 1..4 = VH/retina/ON/choroid
PRED_DIR = "/Users/saurav1/python/masters/arizona/2nd/fall/retina_ultrasound_annotation/work_dir/USFMSuperVit_test_dice0.739/mask_pre"
GT_DIR   = "/Users/saurav1/python/masters/arizona/2nd/fall/retina_ultrasound_annotation/work_dir/USFMSuperVit_test_dice0.739/mask_gt"

per_class_overall = {c: [] for c in range(NUM_CLASSES)}
per_class_present = {c: [] for c in range(NUM_CLASSES)}

def masks_to_logits(pred_mask, num_classes, eps=1e-6):
    # pred_mask: [H,W] int
    t = torch.from_numpy(pred_mask).long()        # [H,W]
    one_hot = torch.nn.functional.one_hot(t, num_classes)  # [H,W,C]
    one_hot = one_hot.permute(2,0,1).unsqueeze(0).float()  # [1,C,H,W]
    # Turn 0/1 probs into logits so Dice code works unchanged
    probs = one_hot * (1 - 2*eps) + eps
    logits = torch.log(probs)
    return logits

for pred_path in sorted(glob.glob(os.path.join(PRED_DIR, "*.png"))):
    name = os.path.basename(pred_path)
    gt_path = os.path.join(GT_DIR, name)

    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    gt   = cv2.imread(gt_path,   cv2.IMREAD_UNCHANGED)

    logits = masks_to_logits(pred, NUM_CLASSES)
    target = torch.from_numpy(gt).unsqueeze(0).long()

    overall_d, present_d, _ = per_class_dice_from_logits(
        logits, target, num_classes=NUM_CLASSES, ignore_index=0
    )

    # accumulate like eval_seg.py
    for c in range(NUM_CLASSES):
        per_class_overall[c].append(float(overall_d.get(c, 0.0)))
        v = present_d.get(c, None)
        per_class_present[c].append(np.nan if v is None else float(v))

# # Then compute FG Dice (overall / present-only) exactly like your eval_seg
# fg_ids = [c for c in range(NUM_CLASSES) if c != 0]
# fg_overall = np.array([np.mean(per_class_overall[c]) for c in fg_ids])
# fg_present = np.array([np.nanmean(per_class_present[c]) for c in fg_ids])


# Then compute FG Dice (overall / present-only) exactly like your eval_seg
for cl in range(1, NUM_CLASSES):
    fg_ids = [c for c in range(NUM_CLASSES) if c == cl]
    fg_overall = np.array([np.mean(per_class_overall[c]) for c in fg_ids])
    fg_present = np.array([np.nanmean(per_class_present[c]) for c in fg_ids])

    print("FG Dice (overall)     :", cl, fg_overall.mean(), "±", fg_overall.std())
    # print("FG Dice (present-only):", cl, np.nanmean(fg_present), "±", np.nanstd(fg_present))