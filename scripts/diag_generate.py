"""Sinh mot bo anh nho de chan doan loi "cac lop sinh ra giong nhau" (docs/investigations/class-collapse.md).

Dung y nguyen GenerateImage/get_pipe/update_pipe cua generate/generate.py, chi doi
2 thu can thu: guidance va cach dat seed.

    python scripts/diag_generate.py <thu muc ra> --dataset eurosat --n 20 \
        --guidance 5 --seed_mode per_class [--lora <thu muc chua pytorch_lora_weights.safetensors>]

--seed_mode shared    : nhu code goc, set_seed(seed) truoc MOI lop -> anh thu k cua
                        moi lop bat dau tu cung mot nhieu
--seed_mode per_class : set_seed(seed * 1000 + vi tri lop) -> moi lop mot nhieu rieng,
                        van tai lap duoc va van chay tiep tung lop duoc
Khong co --lora = SD goc (mode zeroshot). Anh ghi ra <thu muc ra>/train/<lop>/<k>.png,
lop nao da du anh thi bo qua.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "generate"))
from generate import GenerateImage, get_pipe  # noqa: E402
from util import SUBSET_NAMES, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dataset", default="eurosat")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seed_mode", choices=["shared", "per_class"], default="shared")
ap.add_argument("--lora", default=None)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--bs", type=int, default=10)
args = ap.parse_args()

local = yaml.safe_load(open(REPO / "generate" / "local.yaml", encoding="utf-8"))
device = torch.device("cuda")
pipe = get_pipe("sd2.1", local["model_dir"]["sd2.1"], device, is_tqdm=False)
mode = "zeroshot"
if args.lora:
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    mode = "datadream"

gen = GenerateImage(
    pipe=pipe, device=device, mode=mode, guidance_scale=args.guidance,
    num_inference_steps=args.steps, n_img_per_class=args.n, save_dir=args.out,
    count_start=0, bs=args.bs, n_shot=16 if args.lora else 0, n_template=1,
    dataset=args.dataset,
)
for i, cls in enumerate(SUBSET_NAMES[args.dataset]):
    d = Path(args.out) / "train" / cls
    if d.is_dir() and all((d / ("%d.png" % k)).exists() for k in range(args.n)):
        continue
    set_seed(args.seed if args.seed_mode == "shared" else args.seed * 1000 + i)
    gen.run(cls)
    print("xong", cls, flush=True)
