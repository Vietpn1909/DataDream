#!/usr/bin/env python3
"""Tai Stable Diffusion 2.1 base (512) ve may, chi nhung file code can (~5 GB).

    python scripts/download_sd21.py                     # vao ~/models/sd2.1-base
    python scripts/download_sd21.py D:/models/sd2.1-base

Chay giong nhau tren Windows, Linux va Colab.

Vi sao khong dung lenh trong ke hoach ban dau:
  - stabilityai/stable-diffusion-2-1-base da bi Stability AI go khoi
    Hugging Face (API tra ve 401). Dung ban mirror cong dong
    sd2-community/stable-diffusion-2-1-base, cung cau truc diffusers.
  - Tai nguyen repo mirror la ~37 GB vi co nhieu ban trung (.ckpt, .bin, fp16,
    nonema). Code chi can ban fp32 dang .safetensors: ~5 GB.
  - Chi lay .safetensors va .json, KHONG lay .bin/.ckpt: hai dinh dang do la
    pickle, nap vao co the chay code tuy y - khong nen tin tu repo cong dong.

Can huggingface_hub (co trong requirements-gen.txt).
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = "sd2-community/stable-diffusion-2-1-base"

ALLOW = [
    "model_index.json",
    "feature_extractor/*",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

MUST_EXIST = [
    "model_index.json",
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/vocab.json",
    "scheduler/scheduler_config.json",
]


def main():
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "~/models/sd2.1-base")
    dest = dest.expanduser().resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("LOI: chua co huggingface_hub. Chay: pip install -r requirements-gen.txt")

    print("Tai " + REPO + " -> " + str(dest))
    dest.mkdir(parents=True, exist_ok=True)

    # local_dir_use_symlinks=False: ghi file that vao dest thay vi symlink sang
    # cache, de xoa cache sau nay khong lam hong model (va Windows khong can
    # bat Developer Mode cho symlink).
    snapshot_download(
        repo_id=REPO,
        allow_patterns=ALLOW,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )

    print("\nKiem tra:")
    missing = False
    for f in MUST_EXIST:
        p = dest / f
        if p.is_file() and p.stat().st_size > 0:
            print("  ok   %8.2f GB  %s" % (p.stat().st_size / 1e9, f))
        else:
            print("  THIEU            " + f)
            missing = True
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print("  tong: %.2f GB" % (total / 1e9))

    if missing:
        sys.exit("Co file thieu - chay lai script (se tai tiep phan con thieu).")

    print("\nXong. Dien duong dan TUYET DOI nay vao model_dir.sd2.1 trong local.yaml:")
    print("  " + dest.as_posix())


if __name__ == "__main__":
    main()
