"""COCO-json (from ann_v1) + non-nodule PNGs -> YOLO detect dataset (train/valid/test = 70/20/10, like Roboflow default)."""
import json, glob, os, random, sys
S = '/mnt/legend800/lbw_project/lidc_data/tree/root_ndu_images'
O = '/mnt/legend800/lbw_project/lidc_data/yolo_ds'
items = []   # (png, [yolo lines])
for j in sorted(glob.glob(f'{S}/coco_json/*.json')):
    d = json.load(open(j)); im = d['images'][0]; W, H = im['width'], im['height']
    lines = []
    for an in d['annotations']:
        x, y, w, h = an['bbox']
        if w < 2 or h < 2: continue
        lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    if lines: items.append((f"{S}/sop_images/{im['file_name']}", lines))
n_pos = len(items)
for p in sorted(glob.glob(f'{S}/non_nodule_images/*.png')):
    items.append((p, []))
print('positive', n_pos, 'negative', len(items) - n_pos)
random.Random(42).shuffle(items)
n = len(items); cut = {'train': (0, int(n * .7)), 'valid': (int(n * .7), int(n * .9)), 'test': (int(n * .9), n)}
for sp, (a, b) in cut.items():
    os.makedirs(f'{O}/{sp}/images', exist_ok=True); os.makedirs(f'{O}/{sp}/labels', exist_ok=True)
    for png, lines in items[a:b]:
        name = os.path.basename(png)[:-4]
        link = f'{O}/{sp}/images/{name}.png'
        if not os.path.lexists(link): os.symlink(png, link)
        open(f'{O}/{sp}/labels/{name}.txt', 'w').write('\n'.join(lines))
    print(sp, b - a)
open(f'{O}/data.yaml', 'w').write(f"path: {O}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: ['nodule']\n")
