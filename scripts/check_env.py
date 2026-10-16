#!/usr/bin/env python3
"""Kiem tra moi truong va trang thai repo truoc khi chay that.

Chay:  python scripts/check_env.py

Khong import code cua repo (vi nhung file do can torch), chi doc tinh bang
regex, nen chay duoc ca tren Colab lan may chua dung moi truong.
"""
import ast
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
FAIL, WARN = [], []


def ok(msg):
    print("  [ok]    " + msg)


def bad(msg, hint=""):
    print("  [LOI]   " + msg)
    FAIL.append((msg, hint))


def warn(msg, hint=""):
    print("  [canh]  " + msg)
    WARN.append((msg, hint))


def head(title):
    print("\n=== " + title + " ===")


# ---------------------------------------------------------------- torch / GPU
def check_torch():
    head("1. PyTorch va GPU")
    try:
        import torch
    except ImportError:
        warn(
            "chua cai torch",
            "Buoc 1-2: pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128\n"
            "Tren Colab thi torch co san, bo qua muc nay.",
        )
        return
    ok("torch " + torch.__version__)
    if not torch.cuda.is_available():
        warn(
            "torch.cuda.is_available() la False",
            "Binh thuong neu dang o may khong co GPU.",
        )
        return
    name = torch.cuda.get_device_name(0)
    ok("GPU: " + name)
    arch = torch.cuda.get_arch_list()
    is_blackwell = "RTX 50" in name or re.search(r"\b50[6-9]0\b", name)
    if "sm_120" in arch:
        ok("co sm_120 trong arch list (ho tro Blackwell)")
    elif is_blackwell:
        bad(
            "KHONG co sm_120, card la " + name
            + " -> se loi 'no kernel image is available'",
            "Cai lai torch tu index cu128 (xem requirements-gen.txt).",
        )
    else:
        ok("khong co sm_120 nhung card khong phai Blackwell (" + name + ") - bo qua")


# ----------------------------------------------------------------- thu vien
PINS = {
    "diffusers": (
        "0.26",
        "0.29",
        "tu 0.30 mat LoraLoaderMixin va doi chu ky lora_state_dict",
    ),
    "huggingface_hub": (
        "0.20",
        "0.25",
        "tu 0.26 mat huggingface_hub.utils.insecure_hashlib",
    ),
    "transformers": ("4.37", None, "datadream.py yeu cau >= 4.37.2"),
    "accelerate": (None, None, ""),
    "peft": (None, None, ""),
}


def ver_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def check_libs():
    head("2. Thu vien buoc 1-2 (fine-tune + sinh anh)")
    try:
        import diffusers  # noqa: F401
    except ImportError:
        warn(
            "chua cai diffusers",
            "pip install -r requirements-gen.txt "
            "(bo qua neu ban chi lam buoc 3)",
        )
        return

    import importlib.metadata as md

    for pkg, (lo, hi, why) in PINS.items():
        try:
            v = md.version(pkg)
        except md.PackageNotFoundError:
            bad("thieu " + pkg, "pip install -r requirements-gen.txt")
            continue
        t = ver_tuple(v)
        if lo and t < ver_tuple(lo):
            bad(
                pkg + "==" + v + " qua thap (can >= " + lo + "): " + why,
                "pip install -r requirements-gen.txt",
            )
        elif hi and t > ver_tuple(hi):
            bad(
                pkg + "==" + v + " qua cao (can <= " + hi + ".x): " + why,
                "pip install -r requirements-gen.txt",
            )
        else:
            ok(pkg + " " + v)

    # hai import ma code dung va hay vo nhat
    try:
        from diffusers.loaders import LoraLoaderMixin  # noqa: F401

        ok("import duoc diffusers.loaders.LoraLoaderMixin (datadream.py:36)")
    except ImportError:
        bad(
            "KHONG import duoc LoraLoaderMixin -> datadream.py vo ngay dong 36",
            "Ha diffusers ve 0.27.2: pip install -r requirements-gen.txt",
        )
    try:
        from huggingface_hub.utils import insecure_hashlib  # noqa: F401

        ok("import duoc huggingface_hub.utils.insecure_hashlib (datadream.py:51)")
    except ImportError:
        bad(
            "KHONG import duoc insecure_hashlib -> datadream.py vo ngay dong 51",
            "Ha huggingface_hub ve 0.22.2: pip install -r requirements-gen.txt",
        )


# ----------------------------------------------------------------- local.yaml
NEEDED = {
    "DataDream": ["fewshot_data_dir", "model_dir"],
    "generate": ["save_dir", "datadream_dir", "model_dir"],
    "classify": [
        "wandb_key",
        "metadata_dir",
        "clip_download_dir",
        "synth_train_data_dir",
        "real_train_data_dir",
        "real_test_data_dir",
        "real_train_fewshot_data_dir",
    ],
}

PLACEHOLDERS = ("/home/USER", "PATH_TO", "GLOBAL_PATH")


def check_local_yaml():
    head("3. Ba file local.yaml")
    for d, keys in NEEDED.items():
        p = ROOT / d / "local.yaml"
        if not p.exists():
            bad("thieu " + d + "/local.yaml", "bash scripts/setup_local_yaml.sh")
            continue
        text = p.read_text(encoding="utf-8")
        top = set(re.findall(r"^([A-Za-z_]\w*):", text, re.M))
        missing = [k for k in keys if k not in top]
        if missing:
            bad(
                d + "/local.yaml thieu key: " + ", ".join(missing),
                "Copy lai tu configs/local.example.yaml roi dien duong dan.",
            )
        elif any(ph in text for ph in PLACEHOLDERS):
            bad(
                d + "/local.yaml con duong dan mau chua dien",
                "Thay /home/USER/... bang duong dan tuyet doi tren may ban.",
            )
        else:
            ok(d + "/local.yaml du key va da dien")


# ----------------------------------------------------- SUBSET_NAMES 3 module
SUBSET_FILES = {
    "DataDream": ROOT / "DataDream" / "util_data.py",
    "generate": ROOT / "generate" / "util.py",
    "classify": ROOT / "classify" / "util_data.py",
}


def read_subset(path, key="fgvc_aircraft"):
    src = path.read_text(encoding="utf-8")
    m = re.search(r"['\"]" + re.escape(key) + r"['\"]\s*:\s*\[(.*?)\]", src, re.S)
    if not m:
        return None
    try:
        return ast.literal_eval("[" + m.group(1) + "]")
    except (ValueError, SyntaxError):
        return None


def check_classes():
    head("4. Danh sach lop phai khop nhau giua 3 module")
    cfg_path = ROOT / "configs" / "classes_fgvc_aircraft20.txt"
    expected = [
        ln.strip()
        for ln in cfg_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    print("  (configs/classes_fgvc_aircraft20.txt: " + str(len(expected)) + " lop)")

    found = {}
    for name, path in SUBSET_FILES.items():
        lst = read_subset(path)
        if lst is None:
            bad("khong doc duoc SUBSET_NAMES['fgvc_aircraft'] trong " + path.name)
            continue
        found[name] = lst

    if not found:
        return
    if len({tuple(v) for v in found.values()}) > 1:
        bad(
            "SUBSET_NAMES['fgvc_aircraft'] KHAC NHAU giua cac module: "
            + ", ".join(k + "=" + str(len(v)) + " lop" for k, v in found.items()),
            "Ba file phai giong het nhau, neu khong nhan se lech giua "
            "sinh anh va train CLIP.",
        )
        return

    actual = next(iter(found.values()))
    if actual == expected:
        ok("ca 3 module dung dung " + str(len(expected)) + " lop cua configs/")
    else:
        warn(
            "ca 3 module nhat quan nhung dang la " + str(len(actual))
            + " lop, chua cat ve " + str(len(expected)),
            "Chua ap patch cat 20 lop. Xem docs/known-issues.md muc 1.",
        )


# ------------------------------------------------------------- patch da ap?
def check_patches():
    head("5. Cac patch bat buoc")
    gen = (ROOT / "generate" / "generate.py").read_text(encoding="utf-8")

    if re.search(r'assert\s+mode\s*==\s*"datadream"\s+and\s+n_shot\s*>=\s*1', gen):
        bad(
            "generate.py con dong assert loi -> khong sinh duoc anh SD goc "
            "(mode=zeroshot)",
            'Doi thanh: assert mode != "datadream" or n_shot >= 1',
        )
    else:
        ok("da sua dong assert cua mode=zeroshot")

    if re.search(r'revision\s*=\s*"fp16"', gen):
        bad(
            'generate.py con truyen revision="fp16" khi nap model tu thu muc local',
            'Xoa dong revision="fp16" trong get_pipe().',
        )
    else:
        ok('da bo revision="fp16" trong get_pipe()')

    src = (ROOT / "classify" / "data.py").read_text(encoding="utf-8")
    m = re.search(r"real_img_paths\s*=\s*os\.listdir\(.*?\)(.*?)real_subset", src, re.S)
    if m and ".sort(" not in m.group(1):
        bad(
            "classify/data.py chon K-shot khong sort -> classifier train tren "
            "tap K-shot KHAC tap generator da fine-tune",
            "Them real_img_paths.sort(key=natural_keys) ngay sau os.listdir.",
        )
    else:
        ok("classify/data.py da sort truoc khi lay K-shot")


# --------------------------------------------------------------------- main
def main():
    print("Kiem tra repo: " + str(ROOT))
    check_torch()
    check_libs()
    check_local_yaml()
    check_classes()
    check_patches()

    print("\n" + "=" * 64)
    if not FAIL and not WARN:
        print("Tat ca deu on. Chay duoc.")
        return 0
    if FAIL:
        print(str(len(FAIL)) + " loi phai sua:\n")
        for msg, hint in FAIL:
            print("  * " + msg)
            for line in hint.splitlines():
                if line:
                    print("      -> " + line)
    if WARN:
        print("\n" + str(len(WARN)) + " canh bao:\n")
        for msg, hint in WARN:
            print("  * " + msg)
            for line in hint.splitlines():
                if line:
                    print("      -> " + line)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
