#!/usr/bin/env python3
"""Tao tap few-shot EuroSAT tu phan TRAIN cua file chia CoOp.

    python scripts/make_fewshot_eurosat.py                          # 16 anh/lop, seed0
    python scripts/make_fewshot_eurosat.py --shots 16 --seed 1      # them seed1

Dau vao (classify/ cung doc dung cho nay, nen hai buoc dung chung mot ban):
    <root>/train/eurosat/2750/<lop>/<anh>.jpg     torchvision.datasets.EuroSAT tai ve
    <root>/train/split_zhou_EuroSAT.json          file chia cua CoOp
Dau ra:
    <root>/real_train_fewshot/seed<N>/<lop>/<anh>.jpg
    <root>/real_train_fewshot/seed<N>/manifest.json   anh nao duoc chon, chon the nao

Vi sao lay tu phan `train` cua file chia CoOp: classify/ cham diem tren phan
`test` cua chinh file nay. Lay anh few-shot o ngoai phan train la co nguy co anh
dung de fine-tune lot vao tap test. Script tu kiem khong co anh nao nam trong
`val` hay `test`.

Cach chon: voi moi lop, sort ten file trong phan train, roi lay mau ngau nhien
`--shots` anh bang random.Random(seed). Cung seed -> cung tap anh, tren moi may.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT / "data" / "eurosat"))
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ghi de neu thu muc dau ra da co")
    args = ap.parse_args()

    root = Path(args.root)
    split_file = root / "train" / "split_zhou_EuroSAT.json"
    img_root = root / "train" / "eurosat" / "2750"
    out = root / "real_train_fewshot" / ("seed%d" % args.seed)

    if not split_file.is_file():
        sys.exit("LOI: thieu %s - tai tu link trong README cua CoOp (DATASETS.md)." % split_file)
    if not img_root.is_dir():
        sys.exit("LOI: thieu %s - tai bang torchvision.datasets.EuroSAT(root=<root>/train, download=True)." % img_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        sys.exit("LOI: %s da co du lieu. Them --force de tao lai." % out)

    split = json.loads(split_file.read_text(encoding="utf-8"))
    by_class = {}
    for rel, _label, _name in split["train"]:
        by_class.setdefault(rel.split("/")[0], []).append(rel)
    held_out = {rel for part in ("val", "test") for rel, _l, _n in split[part]}

    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)
    manifest = {
        "source": "phan 'train' cua split_zhou_EuroSAT.json (CoOp)",
        "method": "moi lop: sort ten file, random.Random(%d).sample(..., %d), lop theo thu tu ten"
                  % (args.seed, args.shots),
        "seed": args.seed,
        "shots": args.shots,
        "classes": {},
    }
    for cls in sorted(by_class):
        picked = rng.sample(sorted(by_class[cls]), args.shots)
        leak = [p for p in picked if p in held_out]
        if leak:
            sys.exit("LOI: anh %s nam trong val/test - dung lai." % leak)
        (out / cls).mkdir(parents=True)
        for rel in picked:
            src = img_root / rel
            if not src.is_file():
                sys.exit("LOI: khong thay %s" % src)
            shutil.copy2(src, out / cls / src.name)
        manifest["classes"][cls] = sorted(Path(p).name for p in picked)
        print("  %-22s %d anh" % (cls, len(picked)))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nXong: %d lop x %d anh -> %s" % (len(by_class), args.shots, out.as_posix()))
    print("Khong anh nao nam trong phan val/test cua file chia.")


if __name__ == "__main__":
    main()
