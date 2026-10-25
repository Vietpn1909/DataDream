"""Kiem tra mot bo anh sinh truoc khi ban giao: KIEM TOAN BO anh, khong lay mau.

    python scripts/qa_synth.py <thu muc bo anh (chua train/)> <dataset> [--n 200]
        [--after 2026-10-22T22:00] [--out outputs_logs/qa] [--clip D:/models/clip-vit-base-patch16]

In ra tung muc [ok] / [LOI] / [chu y], va ghi luoi anh vao --out:
  <ten bo>_sample.jpg   10 anh ngau nhien moi lop, moi lop mot hang
  <ten bo>_flagged.jpg  cac anh bi danh dau (neu co)

Cac muc kiem:
  1. Cau truc: dung va du thu muc lop (theo SUBSET_NAMES), moi lop dung 0..n-1.png
     + prompts.json, khong co file la (classify coi moi file khong phai .txt/.json la anh)
  2. prompts.json: n prompt, dung template ma generate.py dung
  3. Tung PNG: giai ma duoc toan bo, 512x512 RGB, ghi sau moc --after (khong lan anh cu)
  4. Anh den / trang / gan nhu mot mau
  5. Anh trung: trung tung pixel trong ca bo; gan trung trong cung lop; anh thu k cua cac
     lop khac nhau giong nhau (loi seed chung, known-issues #18)
  6. CLIP ViT-B/16: anh co dung chu de khong (EuroSAT: so voi tam 16 anh that moi lop;
     Aircraft: co phai may bay khong + zero-shot ten lop), do da dang trong lop
"""
import argparse
import datetime
import hashlib
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "generate"))
from util import SUBSET_NAMES, TEMPLATES_SMALL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("dataset")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--after", default="2026-10-22T22:00")
ap.add_argument("--out", default=str(REPO / "outputs_logs" / "qa"))
ap.add_argument("--clip", default="D:/models/clip-vit-base-patch16")
args = ap.parse_args()

root = Path(args.root) / "train"
classes = SUBSET_NAMES[args.dataset]
name = args.dataset + "__" + Path(args.root).name
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
after = datetime.datetime.fromisoformat(args.after).timestamp()
n_err = 0
flagged = []  # (path, ly do)


def ok(msg):
    print("  [ok]     " + msg)


def bad(msg):
    global n_err
    n_err += 1
    print("  [LOI]    " + msg)


def note(msg):
    print("  [chu y]  " + msg)


print("== %s (%s, %d lop x %d anh)" % (args.root, args.dataset, len(classes), args.n))

# ---------------------------------------------------------------- 1. cau truc
dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
extra_dirs = sorted(set(dirs) - set(classes))
missing = [c for c in classes if c not in dirs]
stray = [p.name for p in root.iterdir() if p.is_file()]
if missing or extra_dirs or stray:
    bad("thu muc lop: thieu %s, thua %s, file la o goc %s" % (missing, extra_dirs, stray))
else:
    ok("du %d thu muc lop, ten khop SUBSET_NAMES" % len(classes))

expected_files = {"%d.png" % k for k in range(args.n)} | {"prompts.json"}
struct_bad = []
for c in classes:
    files = {p.name for p in (root / c).iterdir()}
    if files != expected_files:
        struct_bad.append("%s: thieu %d, thua %s" % (c, len(expected_files - files), sorted(files - expected_files)[:5]))
if struct_bad:
    bad("file trong lop: " + "; ".join(struct_bad))
else:
    ok("moi lop dung 0..%d.png + prompts.json, khong co file la" % (args.n - 1))

# ---------------------------------------------------------------- 2. prompts
dsname = {"eurosat": "satellite ", "fgvc_aircraft": "aircraft "}[args.dataset]
pbad = []
for c in classes:
    p = json.load(open(root / c / "prompts.json", encoding="utf-8"))
    want = TEMPLATES_SMALL[0].format(dsname, c)
    if len(p) != args.n or any(x != want for x in p):
        pbad.append(c)
if pbad:
    bad("prompts.json sai o: %s" % pbad)
else:
    ok('prompts.json: %d prompt/lop, dung "%s"' % (args.n, TEMPLATES_SMALL[0].format(dsname, "<lop>")))

# ---------------------------------------------------------------- 3-5. tung anh
paths = [root / c / ("%d.png" % k) for c in classes for k in range(args.n)]
dec_bad, size_bad, old = [], [], []
hashes = defaultdict(list)
gray = np.zeros((len(paths), 32 * 32), dtype=np.float32)
stats = []
for i, p in enumerate(paths):
    try:
        im = Image.open(p)
        im.load()
    except Exception as e:  # noqa: BLE001
        dec_bad.append("%s (%s)" % (p, e))
        continue
    if im.size != (512, 512) or im.mode != "RGB":
        size_bad.append("%s %s %s" % (p.parent.name + "/" + p.name, im.size, im.mode))
    if p.stat().st_mtime < after:
        old.append(p.parent.name + "/" + p.name)
    a = np.asarray(im.convert("RGB"))
    hashes[hashlib.md5(a.tobytes()).hexdigest()].append(p)
    g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    stats.append((float(g.mean()), float(g.std())))
    gray[i] = (g - g.mean()) / (g.std() + 1e-6)

if dec_bad:
    bad("%d anh khong giai ma duoc: %s" % (len(dec_bad), dec_bad[:3]))
else:
    ok("giai ma duoc toan bo %d anh" % len(paths))
if size_bad:
    bad("%d anh sai kich thuoc/mode: %s" % (len(size_bad), size_bad[:3]))
else:
    ok("toan bo 512x512 RGB")
if old:
    bad("%d anh ghi TRUOC moc %s (anh cu sot lai): %s" % (len(old), args.after, old[:5]))
else:
    ok("toan bo anh ghi sau %s (khong lan anh cu)" % args.after)

dark = [paths[i] for i, (m, s) in enumerate(stats) if m < 10]
white = [paths[i] for i, (m, s) in enumerate(stats) if m > 245]
flat = [paths[i] for i, (m, s) in enumerate(stats) if s < 3]
for lst, why in ((dark, "den"), (white, "trang"), (flat, "gan nhu mot mau")):
    flagged += [(p, why) for p in lst]
if dark or white:
    bad("anh den: %d, anh trang: %d" % (len(dark), len(white)))
else:
    ok("khong co anh den / trang")
if flat:
    note("%d anh gan nhu mot mau (do lech chuan < 3/255): %s"
         % (len(flat), sorted({p.parent.name for p in flat})))
else:
    ok("khong co anh gan nhu mot mau")

dups = [v for v in hashes.values() if len(v) > 1]
if dups:
    bad("%d nhom anh TRUNG TUNG PIXEL, vd: %s" % (len(dups), [str(p.parent.name + "/" + p.name) for p in dups[0]]))
    flagged += [(p, "trung pixel") for v in dups for p in v]
else:
    ok("khong co 2 anh nao trung tung pixel trong ca bo")

N, C = args.n, len(classes)
G = gray.reshape(C, N, -1)
near = []
for ci in range(C):
    corr = G[ci] @ G[ci].T / G.shape[2]
    iu = np.triu_indices(N, 1)
    for a_, b_ in zip(*iu):
        if corr[a_, b_] > 0.95:
            near.append((classes[ci], int(a_), int(b_), float(corr[a_, b_])))
if near:
    note("%d cap gan trung (tuong quan pixel > 0.95) trong cung lop, vd %s" % (len(near), near[:3]))
    flagged += [(root / c / ("%d.png" % b_), "gan trung voi %d.png" % a_) for c, a_, b_, _ in near[:20]]
else:
    ok("khong co cap gan trung nao trong cung lop (tuong quan pixel > 0.95)")
same_k = np.array([float(G[i, k] @ G[j, k]) / G.shape[2]
                   for i, j in itertools.combinations(range(C), 2) for k in range(N)])
msg = "anh thu k giua 2 lop khac nhau: tuong quan TB %.3f, %.1f%% cap > 0.8" % (same_k.mean(), 100 * (same_k > 0.8).mean())
(ok if same_k.mean() < 0.1 else bad)(msg + (" (binh thuong ~0)" if same_k.mean() < 0.1 else " -> LOI SEED CHUNG"))

# ---------------------------------------------------------------- 6. CLIP
dev = "cuda"
model = CLIPModel.from_pretrained(args.clip).to(dev).eval()
proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def img_feats(ps):
    fs = []
    for i in range(0, len(ps), 100):
        x = proc(images=[Image.open(p).convert("RGB") for p in ps[i:i + 100]], return_tensors="pt")["pixel_values"].to(dev)
        fs.append(torch.nn.functional.normalize(model.get_image_features(pixel_values=x), dim=-1))
    return torch.cat(fs)


@torch.no_grad()
def txt_feats(t):
    x = proc(text=t, return_tensors="pt", padding=True).to(dev)
    return torch.nn.functional.normalize(model.get_text_features(**x), dim=-1)


F = img_feats(paths).view(C, N, -1)
y = torch.arange(C, device=dev).repeat_interleave(N)
flat_f = F.reshape(C * N, -1)

clip_dup = 0
div = []
for ci in range(C):
    s = F[ci] @ F[ci].T
    clip_dup += int(((s > 0.97).float().triu(1)).sum())
    div.append(float(s[~torch.eye(N, dtype=bool, device=dev)].mean()))
print("  [info]   da dang trong lop: cosine CLIP TB giua 2 anh cung lop = %.3f (thap = da dang); "
      "cap gan trung theo CLIP (> 0.97): %d" % (np.mean(div), clip_dup))

rows = []
if args.dataset == "eurosat":
    fs_dir = REPO / "data/eurosat/real_train_fewshot/seed0"
    cen = torch.nn.functional.normalize(torch.stack(
        [img_feats(sorted((fs_dir / c).iterdir())).mean(0) for c in classes]), dim=-1)
    pred = (flat_f @ cen.T).argmax(1)
    for ci, c in enumerate(classes):
        p = pred[ci * N:(ci + 1) * N].tolist()
        top = max(set(p), key=p.count)
        rows.append((c, p.count(ci), classes[top] if top != ci else "", p.count(top) if top != ci else 0))
    print("  [info]   giong lop that nhat (so voi tam 16 anh that): %.1f%% tong" % (100 * float((pred == y).float().mean())))
else:
    gen = ["a photo of an airplane.", "a photo of an airport terminal building.", "a photo of an empty sky.",
           "a photo of a car.", "a photo of a bird.", "a photo of a person.", "a diagram or text.",
           "a photo of a ship.", "a photo of a toy.", "a photo of a helicopter."]
    pg = (100 * flat_f @ txt_feats(gen).T).softmax(-1)
    not_plane = (pg.argmax(1) != 0).nonzero().flatten().tolist()
    for i in not_plane:
        flagged.append((paths[i], "khong giong may bay: " + gen[int(pg[i].argmax())][len("a photo of "):].rstrip(".")))
    zs = txt_feats(["a photo of a %s, a type of aircraft." % c for c in classes])
    pred = (flat_f @ zs.T).argmax(1)
    for ci, c in enumerate(classes):
        p = pred[ci * N:(ci + 1) * N].tolist()
        top = max(set(p), key=p.count)
        np_c = sum(1 for i in not_plane if i // N == ci)
        rows.append((c, p.count(ci), classes[top] if top != ci else "", p.count(top) if top != ci else 0, np_c))
    msg = "%d/%d anh CLIP khong xep la may bay" % (len(not_plane), len(paths))
    (ok if len(not_plane) < 0.02 * len(paths) else note)(msg)
    print("  [info]   CLIP zero-shot doan dung ten lop: %.1f%% (CLIP chua train, chi de so sanh giua cac lop)"
          % (100 * float((pred == y).float().mean())))

print("  theo lop:")
for r in rows:
    line = "    %-22s dung %3d/%d" % (r[0], r[1], N)
    if r[2]:
        line += " | hay bi xep thanh %s (%d)" % (r[2], r[3])
    if len(r) > 4 and r[4]:
        line += " | %d anh khong giong may bay" % r[4]
    print(line)

# ---------------------------------------------------------------- luoi anh
rng = random.Random(0)
T, LABEL, per = 96, 150, 10
sheet = Image.new("RGB", (LABEL + per * T, C * T), "white")
d = ImageDraw.Draw(sheet)
for r, c in enumerate(classes):
    d.text((4, r * T + T // 2 - 6), c, fill="black")
    for k, idx in enumerate(sorted(rng.sample(range(N), per))):
        sheet.paste(Image.open(root / c / ("%d.png" % idx)).convert("RGB").resize((T, T)), (LABEL + k * T, r * T))
sheet.save(out / (name + "_sample.jpg"), quality=85)
print("  luoi mau: %s" % (out / (name + "_sample.jpg")))
if flagged:
    seen, uniq = set(), []
    for p, why in flagged:
        if p not in seen:
            seen.add(p)
            uniq.append((p, why))
    uniq = uniq[:60]
    cols = 10
    T2 = 128
    fs = Image.new("RGB", (cols * T2, ((len(uniq) + cols - 1) // cols) * (T2 + 24)), "white")
    d = ImageDraw.Draw(fs)
    for i, (p, why) in enumerate(uniq):
        x, yy = (i % cols) * T2, (i // cols) * (T2 + 24)
        fs.paste(Image.open(p).convert("RGB").resize((T2, T2)), (x, yy))
        d.text((x + 2, yy + T2), ("%s/%s" % (p.parent.name, p.name))[:20], fill="black")
        d.text((x + 2, yy + T2 + 11), why[:20], fill="red")
    fs.save(out / (name + "_flagged.jpg"), quality=85)
    print("  anh bi danh dau: %d (luoi %d anh dau: %s)" % (len(seen), len(uniq), out / (name + "_flagged.jpg")))
print("  => %s" % ("DAT" if n_err == 0 else "KHONG DAT: %d loi" % n_err))
