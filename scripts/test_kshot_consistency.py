#!/usr/bin/env python3
"""Kiem tra generator va classifier chon CUNG mot tap K-shot.

Chay:  python scripts/test_kshot_consistency.py

Khong can torch, khong can dataset that - tu dung thu muc tam voi ten file xao
tron, roi chay dung logic chon anh cua hai ben va so ket qua.

Day la bai test cho mot lop loi da that su xay ra: hai ben dung os.listdir ma
mot ben sort, mot ben khong, nen moi may ra mot tap K-shot khac nhau va khong
co gi bao loi. Xem docs/known-issues.md muc 3.
"""
import os
import random
import re
import shutil
import sys
import tempfile
from os.path import join as ospj
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "DataDream"))
sys.path.insert(0, str(ROOT / "classify"))

FAILED = []


def check(cond, label, detail=""):
    print(("  [ok]   " if cond else "  [LOI]  ") + label)
    if not cond:
        if detail:
            print("         " + detail)
        FAILED.append(label)


# ---------------------------------------------------------------------------
# 1. natural_keys cua hai module phai giong het nhau
# ---------------------------------------------------------------------------
def test_natural_keys_identical():
    print("\n1. natural_keys cua DataDream va classify")
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # DataDream/util.py va classify/utils.py deu import torch -> khong exec duoc
    # tren may chua cai. Doc truc tiep source cua hai ham roi exec rieng.
    def extract(path):
        src = Path(path).read_text(encoding="utf-8")
        ns = {"re": re}
        for fn in ("atoi", "natural_keys"):
            m = re.search(
                r"^def " + fn + r"\(.*?(?=^\S|\Z)", src, re.S | re.M
            )
            if not m:
                return None
            exec(m.group(0), ns)
        return ns["natural_keys"]

    nk_gen = extract(ROOT / "DataDream" / "util.py")
    nk_clf = extract(ROOT / "classify" / "utils.py")

    check(nk_gen is not None, "doc duoc natural_keys trong DataDream/util.py")
    check(nk_clf is not None, "doc duoc natural_keys trong classify/utils.py")
    if not (nk_gen and nk_clf):
        return None

    names = ["0.png", "1.png", "2.png", "10.png", "21.png", "100.png",
             "img_3.jpg", "img_12.jpg", "a.JPEG"]
    same = all(nk_gen(n) == nk_clf(n) for n in names)
    check(same, "hai ham cho ket qua giong nhau tren moi ten file thu")
    return nk_gen


# ---------------------------------------------------------------------------
# 2. Hai ben chon cung tap K-shot tu cung mot thu muc
# ---------------------------------------------------------------------------
def test_same_kshot(nk):
    print("\n2. Generator va classifier chon cung tap K-shot")
    tmp = tempfile.mkdtemp()
    try:
        cls_dir = ospj(tmp, "737-300")
        os.makedirs(cls_dir)
        # ten file co so khong dong deu, tao ra dung 16 anh
        fnames = ["%d.png" % i for i in range(16)]
        random.Random(0).shuffle(fnames)
        for f in fnames:
            open(ospj(cls_dir, f), "w").close()

        for n_shot in (1, 4, 16):
            # logic generator: DataDream/datadream.py:214-217
            gen_paths = [ospj(cls_dir, f) for f in os.listdir(cls_dir)]
            gen_paths.sort(key=nk)
            gen = [os.path.basename(p) for p in gen_paths[:n_shot]]

            # logic classifier sau khi patch: classify/data.py:123 va :204
            clf_names = os.listdir(cls_dir)
            clf_names.sort(key=nk)
            clf = [clf_names[i] for i in range(n_shot)]

            check(gen == clf, "K=%-2d hai ben chon cung %d anh" % (n_shot, n_shot),
                  "generator=%s  classifier=%s" % (gen, clf))

        # tap K nho phai la tap con cua tap K lon (can cho ablation K shot)
        paths = [ospj(cls_dir, f) for f in os.listdir(cls_dir)]
        paths.sort(key=nk)
        base = [os.path.basename(p) for p in paths]
        check(base[:1] == base[:16][:1] and base[:4] == base[:16][:4],
              "K=1 va K=4 la tap con long nhau cua K=16")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Chon anh sinh theo n_img_per_cls phai long nhau
# ---------------------------------------------------------------------------
def test_synth_subset(nk):
    print("\n3. Chon anh sinh theo n_img_per_cls phai long nhau")
    tmp = tempfile.mkdtemp()
    try:
        cls_dir = ospj(tmp, "A320")
        os.makedirs(cls_dir)
        fnames = ["%d.png" % i for i in range(200)]
        random.Random(1).shuffle(fnames)
        for f in fnames:
            open(ospj(cls_dir, f), "w").close()
        open(ospj(cls_dir, "prompts.txt"), "w").close()

        def pick(n_img_per_cls):
            out = []
            # logic classify/data.py:187 sau khi patch
            for fname in sorted(os.listdir(cls_dir), key=nk):
                if fname.endswith(".txt") or fname.endswith(".json"):
                    continue
                if len(out) >= n_img_per_cls:
                    break
                out.append(fname)
            return out

        p50, p100, p200 = pick(50), pick(100), pick(200)
        check(len(p50) == 50 and len(p100) == 100 and len(p200) == 200,
              "lay dung so luong 50 / 100 / 200")
        check(p50 == p100[:50], "n=50 la tap con dau cua n=100")
        check(p100 == p200[:100], "n=100 la tap con dau cua n=200")
        check(p200[0] == "0.png" and p200[-1] == "199.png",
              "thu tu la 0..199 (natural sort, khong phai 0,1,10,100)")
        check("prompts.txt" not in p200, "bo qua file .txt")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Chung minh test tren khong rong: bo sort la hai ben lech that
# ---------------------------------------------------------------------------
def test_not_vacuous(nk):
    print("\n4. Khong co sort thi hai ben CO lech (de test tren khong rong)")
    tmp = tempfile.mkdtemp()
    try:
        cls_dir = ospj(tmp, "Spitfire")
        os.makedirs(cls_dir)
        # Tren cung mot thu muc, os.listdir tra ve cung thu tu, nen phai mo
        # phong hai HE THONG TAP TIN khac nhau: do la tinh huong that (may cua
        # Viet vs Colab). Lay chinh thu tu os.listdir lam "may A", va mot thu
        # tu khac lam "may B".
        fnames = ["%d.png" % i for i in range(16)]
        for f in fnames:
            open(ospj(cls_dir, f), "w").close()

        listing_a = os.listdir(cls_dir)
        listing_b = list(reversed(listing_a))  # "may khac", thu tu khac

        # Khong sort: hai may ra hai tap K-shot khac nhau
        no_sort_differ = listing_a[:4] != listing_b[:4]
        check(no_sort_differ,
              "khong sort -> hai may chon tap K-shot KHAC nhau (dung nhu lo)")

        # Co sort: hai may ra cung tap
        a, b = sorted(listing_a, key=nk), sorted(listing_b, key=nk)
        check(a[:4] == b[:4],
              "co sort -> hai may chon CUNG tap K-shot")

        # Va sort theo kieu mac dinh (khong natural) thi thu tu khac natural
        lex = sorted(listing_a)
        check(lex[:4] != a[:4],
              "sorted() thuong khac natural sort -> phai dung dung natural_keys",
              "lexicographic=%s  natural=%s" % (lex[:4], a[:4]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("Test tinh nhat quan K-shot / chon anh sinh")
    print("(Test nay kiem tra LOGIC sort. Viec code co THUC SU goi sort hay")
    print(" khong thi scripts/check_env.py muc 5 kiem tra.)")
    nk = test_natural_keys_identical()
    if nk is None:
        print("\nKhong lay duoc natural_keys, dung lai.")
        return 1
    test_same_kshot(nk)
    test_synth_subset(nk)
    test_not_vacuous(nk)

    print("\n" + "=" * 60)
    if FAILED:
        print("%d test FAIL:" % len(FAILED))
        for f in FAILED:
            print("  * " + f)
        return 1
    print("Tat ca test PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
