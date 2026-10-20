"""Do muc "sap lop" cua cac bo anh chan doan (docs/investigations/class-collapse.md).

    python scripts/diag_eval.py --sets ten=thu_muc [ten=thu_muc ...] [--n 20] \
        [--csv results/diag_class_collapse.csv] [--sheet_dir results/figures]

Moi thu_muc chua train/<lop>/<k>.png (dung k = 0..n-1). Bo "real" (anh that tap
test EuroSAT, khong trung voi tap few-shot) luon duoc do kem de lam moc.

Chi so (CLIP ViT-B/16, cung ban voi buoc classify):
  pix_same_k   tuong quan pixel TB giua anh thu k cua 2 lop KHAC nhau (anh xam 64x64).
               ~0 la binh thuong; cao = cac lop dung chung bo cuc do chung nhieu ban dau
  pix_gt08     % cap (cung k, khac lop) co tuong quan > 0.8 = gan nhu ban sao
  acc_fewshot  % anh duoc gan dung lop khi so voi tam (centroid) CLIP cua 16 anh that
               moi lop - anh sinh co "giong anh that cua dung lop do" khong
  acc_zeroshot % anh CLIP zero-shot doan dung lop qua prompt "a centered satellite photo of ..."
  inter_cos    cosine TB giua tam CLIP cua 2 lop khac nhau trong cung bo; cao = cac lop
               chum lai mot cho
"""
import argparse
import csv
import datetime
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "generate"))
from util import SUBSET_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--clip", default="D:/models/clip-vit-base-patch16")
ap.add_argument("--fewshot", default=str(REPO / "data/eurosat/real_train_fewshot/seed0"))
ap.add_argument("--real_root", default=str(REPO / "data/eurosat/train/eurosat/2750"))
ap.add_argument("--split", default=str(REPO / "data/eurosat/train/split_zhou_EuroSAT.json"))
ap.add_argument("--csv", default=None)
ap.add_argument("--sheet_dir", default=None)
args = ap.parse_args()

classes = SUBSET_NAMES["eurosat"]
split = json.load(open(args.split, encoding="utf-8"))
readable = {}
for rel, _, name in split["test"]:
    readable[rel.split("/")[0]] = name

# bo "real": n anh dau tien moi lop cua tap test (thu tu co dinh theo seed 0)
rng = random.Random(0)
real = {c: [] for c in classes}
for rel, _, _ in split["test"]:
    real[rel.split("/")[0]].append(Path(args.real_root) / rel)
real = {c: rng.sample(sorted(v), args.n) for c, v in real.items()}

sets = {"real": real}
for s in args.sets:
    name, d = s.split("=", 1)
    sets[name] = {c: [Path(d) / "train" / c / ("%d.png" % k) for k in range(args.n)] for c in classes}

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(args.clip).to(dev).eval()
proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def img_feats(paths):
    out = []
    for i in range(0, len(paths), 50):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 50]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        out.append(torch.nn.functional.normalize(model.get_image_features(pixel_values=x), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def text_feats(prompts):
    t = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
    return torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)


fs_paths = {c: sorted((Path(args.fewshot) / c).iterdir()) for c in classes}
centroids = torch.stack([img_feats(fs_paths[c]).mean(0) for c in classes])
centroids = torch.nn.functional.normalize(centroids, dim=-1)
zs = text_feats(["a centered satellite photo of %s." % readable[c] for c in classes])


def gray(p):
    a = np.asarray(Image.open(p).convert("L").resize((64, 64)), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def sheet(name, s, per=8, T=96, LABEL=150):
    im = Image.new("RGB", (LABEL + per * T, len(classes) * T), "white")
    d = ImageDraw.Draw(im)
    for r, c in enumerate(classes):
        d.text((4, r * T + T // 2 - 6), c, fill="black")
        for k in range(per):
            im.paste(Image.open(s[c][k]).convert("RGB").resize((T, T)), (LABEL + k * T, r * T))
    out = Path(args.sheet_dir) / ("diag_eurosat_%s.jpg" % name)
    im.save(out, quality=85)
    return out


rows = []
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
for name, s in sets.items():
    g = {(ci, k): gray(s[c][k]) for ci, c in enumerate(classes) for k in range(args.n)}
    pix = np.array([float(g[(i, k)] @ g[(j, k)]) / 4096
                    for i, j in itertools.combinations(range(len(classes)), 2) for k in range(args.n)])

    f = img_feats([p for c in classes for p in s[c]]).view(len(classes), args.n, -1)
    y = torch.arange(len(classes), device=dev).repeat_interleave(args.n)
    flat = f.reshape(-1, f.shape[-1])
    pred_fs = (flat @ centroids.T).argmax(1)
    pred_zs = (flat @ zs.T).argmax(1)
    cen = torch.nn.functional.normalize(f.mean(1), dim=-1)
    cc = cen @ cen.T
    inter = float(cc[~torch.eye(len(classes), dtype=bool, device=dev)].mean())

    row = dict(set=name, n=args.n,
               pix_same_k=round(float(pix.mean()), 3), pix_gt08=round(100 * float((pix > 0.8).mean()), 1),
               acc_fewshot=round(100 * float((pred_fs == y).float().mean()), 1),
               acc_zeroshot=round(100 * float((pred_zs == y).float().mean()), 1),
               inter_cos=round(inter, 3),
               date=datetime.date.today().isoformat(), commit=commit)
    rows.append(row)
    print("\n== %s" % name)
    print("  " + "  ".join("%s=%s" % (k, row[k]) for k in
                           ("pix_same_k", "pix_gt08", "acc_fewshot", "acc_zeroshot", "inter_cos")))
    # lop nao bi doan sai thanh lop nao (theo tam few-shot)
    for ci, c in enumerate(classes):
        p = pred_fs[ci * args.n:(ci + 1) * args.n].tolist()
        top = max(set(p), key=p.count)
        print("  %-22s dung %2d/%d | hay bi doan thanh: %s (%d)"
              % (c, p.count(ci), args.n, classes[top], p.count(top)))
    if args.sheet_dir:
        print("  luoi anh:", sheet(name, s))

if args.csv:
    new = not Path(args.csv).exists()
    with open(args.csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
