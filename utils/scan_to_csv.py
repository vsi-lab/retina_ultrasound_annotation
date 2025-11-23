#
# import argparse
# import csv
# import random
# from pathlib import Path
#
# import yaml
#
# IMG_EXTS = {'.png','.jpg','.jpeg','.dcm','.dicom'}
# MSK_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff'}
#
# def list_files(d, exts):
#     d = Path(d)
#     return [p for p in d.rglob('*') if p.suffix.lower() in exts]
#
# def stem_of(p: Path):
#     return p.stem
#
# def has_rd_label(mask_path: Path, rd_value):
#     import cv2
#     arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#     return int(arr is not None and (arr == rd_value).any())
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--work_dir', required=False, help='Folder containing images/ and masks/')
#     ap.add_argument('--seed', type=int, default=42)
#     ap.add_argument('--stratify_by_rd', default=True, action='store_true')
#     ap.add_argument('--config', default=None)
#     args = ap.parse_args()
#
#     if args.config:
#         cfg = yaml.safe_load(open(args.config, 'r'))
#     else:
#         raise Exception("missing config file")
#
#     # work_dir = args.work_dir
#     # if work_dir is None:
#     work_dir = cfg['data']['work_dir']
#     work = Path(work_dir)
#
#     imgs_dir = work/cfg['data']['images_sub']
#     msks_dir = work/cfg['data']['masks_sub']
#     out_dir = work/cfg['data']['out_dir']
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     imgs = list_files(imgs_dir, IMG_EXTS)
#     msks = list_files(msks_dir, MSK_EXTS)
#
#     img_map = {stem_of(p): p for p in imgs}
#     msk_map = {stem_of(p): p for p in msks}
#
#     pairs = []
#     for s, ip in img_map.items():
#         if s in msk_map:
#             pairs.append((ip, msk_map[s]))
#     if not pairs:
#         raise SystemExit("No image/mask pairs found. Ensure file stems match.")
#
#     rd_label = cfg['data']['labels']['retinal_detachment']
#
#     random.seed(args.seed)
#     if args.stratify_by_rd:
#         rd_pairs, non_pairs = [], []
#         for ip, mp in pairs:
#             (rd_pairs if has_rd_label(mp, rd_label) else non_pairs).append((ip, mp))
#
#         random.shuffle(rd_pairs); random.shuffle(non_pairs)
#         def split(group, fracs):
#             n = len(group)
#             n_tr = int(n*fracs[0]); n_va = int(n*fracs[1])
#             tr = group[:n_tr]; va = group[n_tr:n_tr+n_va]; te = group[n_tr+n_va:]
#             return tr, va, te
#         tr1, va1, te1 = split(rd_pairs, (cfg['data']['train_frac'], cfg['data']['val_frac'], cfg['data']['test_frac']))
#         tr2, va2, te2 = split(non_pairs, (cfg['data']['train_frac'], cfg['data']['val_frac'],cfg['data']['test_frac']))
#         train, val, test = tr1+tr2, va1+va2, te1+te2
#     else:
#         random.shuffle(pairs)
#         n = len(pairs)
#         n_tr = int(n*cfg['data']['train_frac']); n_va = int(n*acfg['data']['val_frac'])
#         train = pairs[:n_tr]; val = pairs[n_tr:n_tr+n_va]; test = pairs[n_tr+n_va:]
#
#     def write_csv(name, data):
#         f = out_dir/f"{name}.csv"
#         with open(f, 'w', newline='') as fh:
#             w = csv.writer(fh); w.writerow(['image_path','mask_path'])
#             for ip, mp in data:
#                 w.writerow([str(ip), str(mp)])
#         print("Wrote", f)
#
#     write_csv('train', train)
#     write_csv('val', val)
#     write_csv('test', test)
#
# if __name__ == '__main__':
#     main()
