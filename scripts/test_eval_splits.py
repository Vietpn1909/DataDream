"""Kiem tra tap test / tap anh that ma classify/data.py nap (known-issues muc 2 va 17).

    python scripts/test_eval_splits.py [--data D:/DataDream/data]

Dung chinh cac ham trong classify/data.py, khong train gi. Can:
  <data>/eurosat/train/eurosat/2750 + split_zhou_EuroSAT.json  (EuroSAT)
  <data>/eurosat/real_train_fewshot/seed0                      (tap 16-shot)
  <data>/fgvc_aircraft/train/fgvc-aircraft-2013b               (torchvision tu tai)
Thieu dataset nao thi bo qua phan do va bao ra.
"""
import argparse
import collections
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classify"))
try:
    import munch  # noqa: F401
except ImportError:
    # classify/util_data.py import munch cho phan config, cac ham kiem o day khong
    # dung toi -> module gia de chay duoc trong venv sinh anh ma khong cai them.
    import types
    sys.modules["munch"] = types.SimpleNamespace(Munch=dict)
import torchvision as tv  # noqa: E402
from data import filter_aircraft_classes, filter_dset, split_eurosat  # noqa: E402
from util_data import SUBSET_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "data"))
args = ap.parse_args()
fails = []


def check(cond, msg):
    print("  [%s] %s" % ("ok" if cond else "LOI", msg))
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------ EuroSAT
print("== EuroSAT (muc 17)")
root = Path(args.data) / "eurosat" / "train"
if not (root / "eurosat" / "2750").is_dir():
    print("  bo qua: khong thay", root / "eurosat" / "2750")
else:
    def load(split):
        d = tv.datasets.EuroSAT(root=str(root), download=False)
        return split_eurosat(str(root), split, d)

    tr, te = load("train"), load("test")
    trp = {os.path.normpath(s[0]) for s in tr.samples}
    tep = {os.path.normpath(s[0]) for s in te.samples}
    check(len(tr.samples) == 13500, "phan train: %d anh (CoOp: 13.500)" % len(tr.samples))
    check(len(te.samples) == 8100, "tap test: %d anh (CoOp: 8.100)" % len(te.samples))
    check(not (trp & tep), "train va test khong trung anh nao (trung %d)" % len(trp & tep))
    check(list(tr.classes) == SUBSET_NAMES["eurosat"],
          "thu tu lop torchvision == SUBSET_NAMES['eurosat'] (nhan anh that khop nhan anh sinh)")
    cnt = collections.Counter(s[1] for s in te.samples)
    check(sorted(cnt) == list(range(10)), "tap test co du 10 nhan 0..9: %s" % dict(sorted(cnt.items())))
    fs = Path(args.data) / "eurosat" / "real_train_fewshot" / "seed0"
    if fs.is_dir():
        names = {(p.parent.name, p.name) for p in fs.rglob("*.jpg")}
        key = lambda p: (Path(p).parent.name, Path(p).name)  # noqa: E731
        in_tr = sum(key(p) in names for p in trp)
        in_te = sum(key(p) in names for p in tep)
        check(in_tr == len(names) and in_te == 0,
              "%d anh few-shot: %d nam trong phan train, %d nam trong tap test" % (len(names), in_tr, in_te))

# ------------------------------------------------------------------ Aircraft
print("== FGVC-Aircraft (muc 2)")
aroot = Path(args.data) / "fgvc_aircraft" / "train"
if not (aroot / "fgvc-aircraft-2013b").is_dir():
    print("  bo qua: khong thay", aroot / "fgvc-aircraft-2013b")
else:
    keep = SUBSET_NAMES["fgvc_aircraft"]
    for split, per_cls in (("test", (33, 34)), ("trainval", (66, 67))):
        raw = tv.datasets.FGVCAircraft(root=str(aroot), split=split, annotation_level="variant", download=False)
        orig_name = {f: raw.classes[l] for f, l in zip(raw._image_files, raw._labels)}
        n_raw = len(raw._image_files)
        d = filter_aircraft_classes(tv.datasets.FGVCAircraft(
            root=str(aroot), split=split, annotation_level="variant", download=False))
        cnt = collections.Counter(d._labels)
        lo, hi = per_cls
        print("  -- %s: %d anh / %d lop truoc khi loc" % (split, n_raw, len(raw.classes)))
        check(sorted(cnt) == list(range(20)), "con dung 20 nhan 0..19")
        check(d.classes == keep, "d.classes == SUBSET_NAMES['fgvc_aircraft']")
        check(all(lo <= c <= hi for c in cnt.values()),
              "moi lop %s anh: %s" % ("-".join(map(str, per_cls)), sorted(set(cnt.values()))))
        wrong = sum(orig_name[f] != keep[l] for f, l in zip(d._image_files, d._labels))
        check(wrong == 0, "nhan moi tro dung ten lop goc cua tung anh (sai %d/%d)" % (wrong, len(d._labels)))
        if split == "trainval":
            a = filter_dset(d, 16, "fgvc_aircraft")._image_files
            b = filter_dset(filter_aircraft_classes(tv.datasets.FGVCAircraft(
                root=str(aroot), split=split, annotation_level="variant", download=False)),
                16, "fgvc_aircraft")._image_files
            check(len(a) == 320 and a == b, "filter_dset 16 anh/lop: %d anh, 2 lan chay ra cung mot tap" % len(a))

# ------------------------------------------- qua get_data_loader (main.py dung ham nay)
print("== get_data_loader: dung dung ham ma main.py goi")
from data import get_data_loader  # noqa: E402
for ds, sub, n_test, n_cls in (("eurosat", "eurosat", 8100, 10), ("fgvc_aircraft", "fgvc_aircraft", 668, 20)):
    rd = Path(args.data) / sub
    if not (rd / "train").is_dir():
        print("  bo qua", ds)
        continue
    tr, te = get_data_loader(real_train_data_dir=str(rd), dataset=ds, is_synth_train=False,
                             n_img_per_cls=16, model_type="clip")
    te_lab = te.dataset.targets if ds == "eurosat" else te.dataset._labels
    tr_lab = tr.dataset.targets if ds == "eurosat" else tr.dataset._labels
    check(len(te.dataset) == n_test and sorted(set(te_lab)) == list(range(n_cls)),
          "%s tap test: %d anh, nhan 0..%d" % (ds, len(te.dataset), max(te_lab)))
    check(len(tr.dataset) == 16 * n_cls and sorted(set(tr_lab)) == list(range(n_cls)),
          "%s anh that (n_img_per_cls=16): %d anh, nhan 0..%d" % (ds, len(tr.dataset), max(tr_lab)))
    if ds == "eurosat":
        a = {os.path.normpath(s[0]) for s in tr.dataset.samples}
        b = {os.path.normpath(s[0]) for s in te.dataset.samples}
        check(not (a & b), "eurosat: anh that dung de train khong nam trong tap test")

print("\nKET QUA:", "DAT" if not fails else "KHONG DAT - %d loi" % len(fails))
sys.exit(1 if fails else 0)
