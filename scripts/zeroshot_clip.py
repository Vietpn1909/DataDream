#!/usr/bin/env python3
"""Do CLIP zero-shot (chua train) tren DUNG tap test va tien xu ly ma classify/ dung.

    python scripts/zeroshot_clip.py --dataset eurosat       [--csv results/clf_zeroshot.csv]
    python scripts/zeroshot_clip.py --dataset fgvc_aircraft [--csv results/clf_zeroshot.csv]

Dung lai code cua classify/, khong viet lai:
  - tap test: data.get_data_loader(...) -> test_loader: cung anh, cung thu tu nhan,
    cung test_transform voi luc train CLIP
  - model: models.clip.CLIP(..., is_lora_image=False, is_lora_text=False) - chinh model
    ma classify bat dau train, tuc accuracy cua classifier truoc buoc train dau tien
Duong dan doc tu classify/local.yaml (real_train_data_dir, clip_download_dir...).

Hai bo prompt, bao ca hai:
  classify   model.tokenized_text: TEMPLATES_SMALL[0] + ten mien, vd
             "a aircraft photo of a 737-300", "a satellite photo of a AnnualCrop"
             -> dung dau phan loai ma classify bat dau train (bo chinh)
  reference  template zero-shot cua CoOp (trainers/zsclip.py):
             "a photo of a {}, a type of aircraft." / "a centered satellite photo of {}."
             EuroSAT dung ten de doc o cot 3 cua split_zhou_EuroSAT.json ("Annual Crop Land")

In ra: so anh test (tong, tung lop), top-1 kem khoang tin cay Wilson 95%, accuracy
tung lop, 3 cap nham nhieu nhat. --csv: append 2 dong (moi bo prompt mot dong) theo
dung header cua results/results_template.csv.

Aircraft o day la Aircraft-20 (20 lop, doan bua = 5%): KHONG so voi 23,8% cua paper
(100 lop).
"""
import argparse
import collections
import csv
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
CLF = REPO / "classify"
N_TEST = {"eurosat": 8100, "fgvc_aircraft": 668}  # docs/known-issues.md muc 17 va 2
LABEL = {"eurosat": "EuroSAT", "fgvc_aircraft": "Aircraft-20"}
RUN_PREFIX = {"eurosat": "eurosat", "fgvc_aircraft": "aircraft20"}  # cung tien to voi ten zip ban giao
REFERENCE = {"eurosat": "a centered satellite photo of {}.",
             "fgvc_aircraft": "a photo of a {}, a type of aircraft."}


def wilson(k, n, z=1.96):
    """Khoang tin cay Wilson cho ti le k/n (cong thuc dong, khong can thu vien)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(N_TEST))
    ap.add_argument("--csv", default=None, help="vd results/clf_zeroshot.csv; bo trong thi chi in ra")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    csv_path = Path(args.csv).resolve() if args.csv else None  # tinh truoc khi chdir

    # classify/ mo local.yaml (va metadata_dir) bang duong dan tuong doi.
    os.chdir(CLF)
    sys.path.insert(0, str(CLF))
    import clip
    from data import get_data_loader
    from models.clip import CLIP
    from util_data import SUBSET_NAMES

    with open("local.yaml", encoding="utf-8") as f:
        local = yaml.safe_load(f)
    classes = SUBSET_NAMES[args.dataset]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Nap model truoc: nhanh EuroSAT cua get_data_loader tat kiem tra SSL cho ca tien trinh.
    model = CLIP(args.dataset, is_lora_image=False, is_lora_text=False,
                 clip_download_dir=local["clip_download_dir"], clip_version="ViT-B/16").eval().to(dev)

    # is_synth_train=True chi la cach co san de get_data_loader KHONG dung train loader
    # (tra ve None); nhanh tap test van chay y nhu luc train CLIP.
    _, test_loader = get_data_loader(
        real_train_data_dir=local["real_train_data_dir"][args.dataset],
        real_test_data_dir=local["real_test_data_dir"][args.dataset],
        metadata_dir=local["metadata_dir"], dataset=args.dataset,
        is_synth_train=True, model_type="clip", eval_bs=args.bs)
    test_set = test_loader.dataset
    if len(test_set) != N_TEST[args.dataset]:
        sys.exit("LOI: tap test co %d anh, can %d - kiem du lieu bang scripts/test_eval_splits.py"
                 % (len(test_set), N_TEST[args.dataset]))
    # Cung dataset, shuffle=False -> cung anh, cung thu tu. Chi doi so worker: test_transform
    # cua classify chua lambda, Windows (spawn) khong pickle duoc sang tien trinh con.
    loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False,
                                         num_workers=0 if os.name == "nt" else min(4, os.cpu_count() or 1))

    if args.dataset == "eurosat":
        # cung file chia ma split_eurosat() doc, nhu scripts/diag_eval.py
        with open(Path(local["real_train_data_dir"]["eurosat"]) / "train" / "split_zhou_EuroSAT.json",
                  encoding="utf-8") as f:
            readable = {rel.split("/")[0]: name for rel, _, name in json.load(f)["test"]}
        ref_names = [readable[c] for c in classes]
    else:
        ref_names = classes
    prompts = {
        "classify": [model.templates[0].format(model.dataset_name, c) for c in classes],
        "reference": [REFERENCE[args.dataset].format(c) for c in ref_names],
    }
    tokens = {
        "classify": model.tokenized_text,  # dung tensor classify dung, khong tokenize lai
        "reference": clip.tokenize(prompts["reference"]).unsqueeze(1).to(dev),
    }

    # classify/main.py eval() chay trong autocast fp16 (--use_fp16 mac dinh True); lam y vay.
    labels, preds = [], {k: [] for k in tokens}
    with torch.no_grad(), torch.autocast("cuda", enabled=(dev == "cuda")):
        text = {k: model.forward_text(t) for k, t in tokens.items()}
        scale = model.clip.logit_scale.exp()
        for x, y in loader:
            feats = model.forward_image(x.to(dev))
            for k in text:
                preds[k].append((scale * feats @ text[k].t()).argmax(1).cpu())
            labels.append(y)
    y = torch.cat(labels)
    preds = {k: torch.cat(v) for k, v in preds.items()}
    n = len(y)

    print("\n== %s: %d anh test, %d lop | CLIP ViT-B/16 chua train, %s" % (LABEL[args.dataset], n, len(classes), dev))
    for k in prompts:
        print('  prompt %-9s vd "%s"' % (k, prompts[k][0]))
    print("\n  %-22s %5s %10s %10s" % ("lop", "n", "classify", "reference"))
    for i, c in enumerate(classes):
        m = y == i
        accs = [100 * float((preds[k][m] == i).float().mean()) for k in ("classify", "reference")]
        print("  %-22s %5d %9.1f%% %9.1f%%" % (c, int(m.sum()), accs[0], accs[1]))

    results = {}
    for k, p in preds.items():
        correct = int((p == y).sum())
        lo, hi = wilson(correct, n)
        acc, lo, hi = 100 * correct / n, 100 * lo, 100 * hi
        results[k] = (acc, lo, hi, correct)
        print("\n  %-9s top-1 %.2f%% (%d/%d), 95%% Wilson [%.2f, %.2f]" % (k, acc, correct, n, lo, hi))
        conf = collections.Counter((int(t), int(q)) for t, q in zip(y, p) if t != q).most_common(3)
        print("            nham nhieu nhat: " + "; ".join(
            "%s -> %s (%d anh)" % (classes[t], classes[q], cnt) for (t, q), cnt in conf))

    if csv_path:
        with open(REPO / "results" / "results_template.csv", encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        if csv_path.exists():
            with open(csv_path, encoding="utf-8-sig") as f:
                if next(csv.reader(f), None) != header:
                    sys.exit("LOI: header cua %s khac results/results_template.csv" % csv_path)
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new:
                w.writeheader()
            for k, (acc, lo, hi, correct) in results.items():
                w.writerow({
                    "run_name": "%s_zeroshot_%s" % (RUN_PREFIX[args.dataset], k),
                    "dataset": args.dataset, "variant": "none", "K": 0, "n_img": 0, "seed": "",
                    "setting": "clip-zeroshot-%s" % k, "acc": "%.2f" % acc,
                    "date": datetime.date.today().isoformat(), "commit": commit,
                    "note": "%s; Wilson95 %.2f-%.2f; %d/%d anh test; ViT-B/16 chua train; prompt: %s"
                            % (LABEL[args.dataset], lo, hi, correct, n, prompts[k][0]),
                })
        print("\nDa ghi 2 dong vao %s" % csv_path)


if __name__ == "__main__":
    main()
