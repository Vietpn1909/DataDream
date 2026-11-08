#!/usr/bin/env python3
"""Tao tap few-shot FGVC-Aircraft (20 lop) tu phan TRAINVAL cua bo du lieu goc.

    python scripts/make_fewshot_aircraft.py --download              # tai dataset neu chua co (~2,7 GB)
    python scripts/make_fewshot_aircraft.py                         # 16 anh/lop, seed0
    python scripts/make_fewshot_aircraft.py --shots 16 --seed 1     # them seed1

Dau vao (classify/ cung doc dung cho nay, nen hai buoc dung chung mot ban):
    <root>/train/fgvc-aircraft-2013b/data/variants.txt
    <root>/train/fgvc-aircraft-2013b/data/images_variant_trainval.txt
    <root>/train/fgvc-aircraft-2013b/data/images_variant_test.txt
    <root>/train/fgvc-aircraft-2013b/data/images/<image_id>.jpg
Dau ra:
    <root>/real_train_fewshot/seed<N>/<lop>/01.jpg ... 16.jpg
    <root>/real_train_fewshot/seed<N>/manifest.json   anh nao duoc chon, chon the nao

Vi sao chi lay tu `trainval`: classify/ cham diem tren phan `test`
(torchvision FGVCAircraft(split='test')). Lay anh few-shot o ngoai trainval la co
nguy co anh dung de fine-tune lot vao tap test. Script tu kiem khong co anh nao
nam trong `test`.

Cach chon: MOT random.Random(seed) cho ca 20 lop. Di qua cac lop theo thu tu
configs/classes_fgvc_aircraft20.txt; moi lop sort image id phan trainval, roi
rng.sample(..., --shots). File dat ten 01.jpg, 02.jpg... theo THU TU BOC: DataDream/
va classify/ sort ten file bang natural_keys roi lay K anh dau, nen anh boc dau
tien la tap 1-shot, 4 anh dau la tap 4-shot. Cung seed -> cung tap anh, tren moi may.

Banner ban quyen (~20 px o day moi anh): mac dinh GIU NGUYEN, chep file y nguyen
tung byte. classify/ nap tap test qua torchvision ma khong cat banner, nen chi
cat o tap few-shot thi anh train va anh test lech nhau. --crop_banner cat 20 px
duoi roi luu JPEG chat luong 95 - chi dung khi ca nhom da thong nhat cat o moi noi.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "DataDream"))
from util import natural_keys  # noqa: E402  - DataDream/util.py, ham datadream.py dung de lay K anh

BANNER_PX = 20


def subset_names_generate():
    # generate/util.py trung ten module `util` voi DataDream/util.py o tren, nen
    # nap no duoi mot ten khac.
    spec = importlib.util.spec_from_file_location("generate_util", ROOT / "generate" / "util.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.SUBSET_NAMES["fgvc_aircraft"])


def read_split(path):
    # Moi dong: "<image_id> <ten variant>". Ten variant co the co dau cach
    # ("Cessna 172"), nen chi tach o dau cach DAU TIEN - y nhu torchvision
    # FGVCAircraft (classify/ doc nhan bang cach do).
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            image_id, name = line.strip().split(" ", 1)
            pairs.append((image_id, name))
    return pairs


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT / "data" / "fgvc_aircraft"))
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ghi de neu thu muc dau ra da co")
    ap.add_argument("--download", action="store_true",
                    help="tai FGVC-Aircraft qua torchvision neu chua co (~2,7 GB)")
    ap.add_argument("--crop_banner", action="store_true",
                    help="cat %d px banner ban quyen o day anh (mac dinh: giu nguyen)" % BANNER_PX)
    args = ap.parse_args()

    root = Path(args.root)
    data = root / "train" / "fgvc-aircraft-2013b" / "data"
    out = root / "real_train_fewshot" / ("seed%d" % args.seed)

    if args.download:
        import torchvision as tv
        # Dung loi goi va thu muc ma classify/data.py dung -> hai buoc dung chung mot ban.
        tv.datasets.FGVCAircraft(root=str(root / "train"), split="trainval",
                                 annotation_level="variant", download=True)
    if not (data / "images_variant_trainval.txt").is_file():
        sys.exit("LOI: thieu %s - them --download, hoac tai bang "
                 "torchvision.datasets.FGVCAircraft(root=<root>/train, download=True)." % data)
    if out.exists() and any(out.iterdir()) and not args.force:
        sys.exit("LOI: %s da co du lieu. Them --force de tao lai." % out)

    cfg = ROOT / "configs" / "classes_fgvc_aircraft20.txt"
    classes = [ln.strip() for ln in cfg.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    if classes != subset_names_generate():
        sys.exit("LOI: %s khac SUBSET_NAMES['fgvc_aircraft'] cua generate/util.py (ten hoac thu tu)." % cfg.name)
    variants = {ln.strip() for ln in (data / "variants.txt").read_text(encoding="utf-8").splitlines()}
    missing = [c for c in classes if c not in variants]
    if missing:
        sys.exit("LOI: lop khong co nguyen van trong variants.txt: %s" % missing)

    trainval = read_split(data / "images_variant_trainval.txt")
    test_ids = {i for i, _ in read_split(data / "images_variant_test.txt")}
    overlap = {i for i, _ in trainval} & test_ids
    if overlap:
        sys.exit("LOI: %d anh nam o ca trainval lan test, vd %s - dung lai." % (len(overlap), sorted(overlap)[:3]))
    by_class = {}
    for image_id, name in trainval:
        by_class.setdefault(name, []).append(image_id)

    # NHOM SUA (review PR #6): boc va kiem HET truoc (du anh, khong lot tap test, co file),
    # roi moi xoa tap cu - truoc day --force xoa seed<N> roi moi phat hien loi, mat tap dang
    # dung de fine-tune LoRA. Thu tu goi rng y nhu cu nen ket qua khong doi.
    rng = random.Random(args.seed)
    picks = {}
    for cls in classes:
        candidates = sorted(by_class.get(cls, []))
        if len(candidates) < args.shots:
            sys.exit("LOI: lop %s chi co %d anh trainval, can %d." % (cls, len(candidates), args.shots))
        picks[cls] = (rng.sample(candidates, args.shots), len(candidates))
        leak = [i for i in picks[cls][0] if i in test_ids]
        if leak:
            sys.exit("LOI: anh %s nam trong tap test - dung lai." % leak)
        missing_src = [i for i in picks[cls][0] if not (data / "images" / (i + ".jpg")).is_file()]
        if missing_src:
            sys.exit("LOI: thieu anh goc %s - them --download hoac tai lai dataset." % missing_src[:3])
    if out.exists():
        shutil.rmtree(out)
    manifest = {
        "source": "phan 'trainval' cua fgvc-aircraft-2013b (data/images_variant_trainval.txt)",
        "method": "MOT random.Random(%d) cho ca %d lop, theo thu tu configs/classes_fgvc_aircraft20.txt; "
                  "moi lop: sort image id phan trainval, rng.sample(..., %d); file 01..%02d.jpg theo thu tu boc"
                  % (args.seed, len(classes), args.shots, args.shots),
        "seed": args.seed,
        "shots": args.shots,
        "crop_banner": args.crop_banner,
        "class_order": classes,
        "classes": {},
    }
    for cls in classes:
        picked, n_candidates = picks[cls]
        (out / cls).mkdir(parents=True)
        files = []
        for k, image_id in enumerate(picked, 1):
            src = data / "images" / (image_id + ".jpg")
            dst = out / cls / ("%02d.jpg" % k)
            if args.crop_banner:
                im = Image.open(src)
                im.crop((0, 0, im.width, im.height - BANNER_PX)).save(dst, quality=95)
            else:
                shutil.copy2(src, dst)
            files.append({"file": dst.name, "image_id": image_id, "sha256": sha256(dst)})
        manifest["classes"][cls] = files
        print("  %-12s %d anh (trong %d anh trainval)" % (cls, len(picked), n_candidates))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Tu kiem: thu tu ma DataDream/ va classify/ se doc phai dung la thu tu boc.
    want = ["%02d.jpg" % k for k in range(1, args.shots + 1)]
    for cls in classes:
        got = sorted(os.listdir(out / cls), key=natural_keys)
        if got != want:
            sys.exit("LOI: %s/%s sort ra %s, can %s" % (out, cls, got[:3], want[:3]))

    n = len(classes) * args.shots
    print("\nXong: %d lop x %d anh = %d anh -> %s" % (len(classes), args.shots, n, out.as_posix()))
    print("Trung voi tap test (images_variant_test.txt): 0/%d anh; trainval va test khong giao nhau." % n)
    print("Moi lop sort natural_keys ra dung 01..%02d.jpg (= thu tu boc)." % args.shots)
    print("Banner ban quyen: %s." % ("DA CAT %d px duoi, luu JPEG q95" % BANNER_PX if args.crop_banner
                                     else "giu nguyen, chep y nguyen tung byte"))


if __name__ == "__main__":
    main()
