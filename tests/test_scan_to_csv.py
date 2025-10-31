# # tests/test_scan_to_csv.py
# import numpy as np, cv2, pandas as pd, yaml
# from utils.scan_to_csv import scan_and_split

# from pathlib import Path

# CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

# def test_scan_stratify_rd(tmp_path):
#     cfg = yaml.safe_load(open(CFG_PATH, 'r'))
#     d = tmp_path/'work_dir'
#     (d/'images').mkdir(parents=True); (d/'masks').mkdir()
#     # create 4 images, 2 with RD pixels
#     lbl_rd = cfg['data']['labels']['retinal_detachment']
#     for i in range(4):
#         img = np.zeros((64,64), np.uint8); cv2.circle(img, (20,20), 10, 200, -1)
#         msk = np.zeros((64,64), np.uint8)
#         if i%2==0: msk[10:20,10:20] = lbl_rd
#         cv2.imwrite(str(d/'images'/f'i{i}.png'), img)
#         cv2.imwrite(str(d/'masks'/f'i{i}.png'), msk)
#
#     out_dir = d/'data'; out_dir.mkdir()
#     train_csv, val_csv, test_csv = scan_and_split(d, cfg, out_dir)
#
#     # ensure both strata appear across splits (best-effort with tiny set)
#     import pandas as pd
#     for p in [train_csv, val_csv, test_csv]:
#         df = pd.read_csv(p)
#         has_rd = False; has_non = False
#         for _, r in df.iterrows():
#             m = cv2.imread(r['mask_path'], cv2.IMREAD_GRAYSCALE)
#             if (m == lbl_rd).any(): has_rd = True
#             else: has_non = True
#         assert has_rd and has_non, f"Strata missing in split {p}"
