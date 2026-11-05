# NHOM THEM: file moi - sinh bo anh baseline IsSynth (He et al., ICLR 2023), "real-data
# guidance", de so voi bo SD goc va DataDream. Khong phai code goc cua tac gia DataDream.
"""Sinh bo anh IsSynth (real-data guidance) bang SD 2.1 img2img.

    cd generate
    python issynth.py --dataset=fgvc_aircraft --n_img_per_class=10 --strength=0.15
    python issynth.py --dataset=fgvc_aircraft --n_img_per_class=5 --strength=0.3 --classes 2 10 16 18 \
        --save_dir=../outputs_logs/issynth/sweep

IsSynth trong setting few-shot: lay mot anh that few-shot, them nhieu toi muc --strength,
roi khu nhieu bang SD 2.1 voi prompt cua lop -> anh sinh bam sat phan phoi anh that.
Paper DataDream chay lai IsSynth bang SD 2.1, prompt "a photo of a [CLS]", 500 anh/lop,
KHONG loc. O day cung KHONG lam Language Enhancement (cau T5) va bo loc CLIP cua IsSynth
goc - DataDream bo ca hai.

Cong bang voi bo SD goc va DataDream: CHI khac o real guidance. Giu nguyen model, prompt
(TEMPLATES_SMALL[0] + ten mien, nhu generate.py), guidance, so buoc, so anh/lop, seed tung
lop (seed*1000 + vi tri lop trong SUBSET_NAMES), anh PNG 512x512, va dung 16 anh that ma
LoRA da fine-tune. Anh thu k cua lop lay anh that refs[k % n_shot], voi refs = n_shot anh
dau sau khi sort natural_keys (cung cach DataDream/ va classify/ chon K anh).

Dau ra (cung kieu duong dan voi generate.py, them hau to _issynth_str<strength>):
  <save_dir>/<dataset>/sd2.1/gs2.0_nis50/shot16_seed0_template1_issynth_str0.15/
      train/<lop>/0.png ... (n-1).png, prompts.json
      issynth_refs.json   {lop: [ten file anh that dung cho anh k = 0..n-1]} - o goc bo,
                          KHONG nam trong train/ (qa_synth.py bao loi neu co file la o do)

Chay lai bao nhieu lan cung duoc: lop nao da dung 0..n-1.png va doc duoc het thi bo qua;
lop dang do thi xoa PNG cua lop do va sinh lai tu k=0 - khong bao gio sinh tiep giua
chung (docs/known-issues.md muc 9).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from os.path import join as ospj

import torch
import torchvision as tv
import yaml
from PIL import Image, ImageOps

from generate import get_dataset_name_for_template
from util import SUBSET_NAMES, TEMPLATES_SMALL, batch_iteration, make_dirs, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))

# natural_keys cua DataDream/util.py - ham datadream.py dung de chon K anh that. File do
# trung ten module `util` voi generate/util.py, nen nap duoi ten khac.
_spec = importlib.util.spec_from_file_location("datadream_util", ospj(HERE, "..", "DataDream", "util.py"))
_datadream_util = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_datadream_util)
natural_keys = _datadream_util.natural_keys

# Anh that: canh ngan ve 512 (bicubic) roi cat giua 512x512. Khong cat banner o day:
# quyet dinh ve banner (D1) da ap san len tap few-shot.
REF_TRANSFORM = tv.transforms.Compose([
    tv.transforms.Resize(512, interpolation=tv.transforms.InterpolationMode.BICUBIC),
    tv.transforms.CenterCrop(512),
])


def load_ref(path):
    return REF_TRANSFORM(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def save_data(outputs, save_dir, count):
    # giong het GenerateImage.save_data trong generate.py
    for image in outputs.images:
        image.resize((512, 512)).save(ospj(save_dir, f"{count}.png"))
        count += 1
    return count


def class_done(cls_dir, n):
    """Lop da xong = co prompts.json, dung 0..n-1.png (khong thua khong thieu), doc duoc het."""
    if not os.path.isfile(ospj(cls_dir, "prompts.json")):
        return False
    pngs = {f for f in os.listdir(cls_dir) if f.endswith(".png")}
    if pngs != {f"{k}.png" for k in range(n)}:
        return False
    try:
        for f in pngs:
            with Image.open(ospj(cls_dir, f)) as im:
                im.load()
    except Exception:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="fgvc_aircraft")
    ap.add_argument("--strength", type=float, default=0.15,
                    help="muc nhieu them vao anh that, 0..1 (D2: chot truoc khi train classifier)")
    ap.add_argument("--n_img_per_class", type=int, default=200)
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="vi tri lop trong SUBSET_NAMES; bo trong = moi lop")
    ap.add_argument("--sd_version", default="sd2.1")
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--n_shot", type=int, default=16)
    ap.add_argument("--n_template", type=int, default=1)
    ap.add_argument("--fewshot_seed", default="seed0")
    ap.add_argument("--fewshot_dir", default=None, help="ghi de fewshot_data_dir[dataset] cua local.yaml")
    ap.add_argument("--save_dir", default=None, help="ghi de save_dir cua local.yaml (vd cho lan chay thu)")
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # local.yaml o thu muc hien tai, nhu generate.py. Khai bao utf-8: file mau co comment tieng
    # Viet, ma generate.set_local() mo khong encoding -> Windows dung code page cp1252 bao
    # UnicodeDecodeError.
    with open("local.yaml", encoding="utf-8") as f:
        args_local = yaml.safe_load(f)
    model_dir = args_local["model_dir"][args.sd_version]
    save_root = args.save_dir or args_local["save_dir"]
    fewshot_root = args.fewshot_dir or args_local.get("fewshot_data_dir", {}).get(args.dataset)
    if not fewshot_root:
        sys.exit("LOI: local.yaml khong co fewshot_data_dir.%s - truyen --fewshot_dir" % args.dataset)
    fewshot_dir = ospj(fewshot_root, args.fewshot_seed)

    set_dir = ospj(
        save_root, args.dataset, args.sd_version,
        f"gs{args.guidance_scale}_nis{args.num_inference_steps}",
        f"shot{args.n_shot}_{args.fewshot_seed}_template{args.n_template}_issynth_str{args.strength}",
    )
    print(set_dir)

    all_classes = SUBSET_NAMES[args.dataset]
    todo = all_classes if not args.classes else [all_classes[i] for i in args.classes]

    # Kiem anh that cua moi lop TRUOC khi nap model.
    refs = {}
    for classname in todo:
        d = ospj(fewshot_dir, classname)
        names = sorted(os.listdir(d), key=natural_keys)[: args.n_shot]
        if len(names) < args.n_shot:
            sys.exit("LOI: %s chi co %d anh, can %d" % (d, len(names), args.n_shot))
        refs[classname] = names

    # prompt: y het GenerateImage.name_template_method cua generate.py
    dataset_name = get_dataset_name_for_template(args.dataset)
    templates = TEMPLATES_SMALL[: args.n_template]
    n = args.n_img_per_class
    n_repeat = n // len(templates) + 1

    from diffusers import StableDiffusionImg2ImgPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cung thu muc weights voi generate.py (va cung scheduler trong do)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # VAE ma hoa / giai ma tung anh mot thay vi ca lo 10 anh 512x512: cung phep tinh, cung so
    # lan rut ngau nhien (posterior van lay mau mot lan cho ca lo), chi ha dinh VRAM. Khong co
    # dong nay, bs=10 can ~8 GB -> card 6 GB tran sang RAM, cham ~10 lan (known-issues muc 10).
    pipe.enable_vae_slicing()

    timesteps = []  # timestep cua tung buoc khu nhieu, ghi o lan goi pipe dau tien

    def log_step(pipeline, step, t, kwargs):
        if len(timesteps) == step:  # cac lan goi sau: step < len(timesteps)
            timesteps.append(int(t))
        return kwargs

    # Anh that cua anh k la co dinh (refs[k % n_shot]) nen ghi map truoc khi sinh. Chay
    # --classes thi gop vao file da co.
    refs_path = ospj(set_dir, "issynth_refs.json")
    refs_log = {}
    if os.path.isfile(refs_path):
        with open(refs_path, encoding="utf-8") as f:
            refs_log = json.load(f)
    for classname in todo:
        refs_log[classname] = [refs[classname][k % args.n_shot] for k in range(n)]
    os.makedirs(set_dir, exist_ok=True)
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump(refs_log, f, indent=2, ensure_ascii=False)

    t_total = time.time()
    n_new = 0
    for classname in todo:
        cls_dir = ospj(set_dir, "train", classname)
        if class_done(cls_dir, n):
            print("  %-12s bo qua - da du %d anh" % (classname, n))
            continue
        make_dirs(cls_dir)
        for f in os.listdir(cls_dir):  # lop dang do: xoa het PNG, sinh lai tu k=0
            if f.endswith(".png"):
                os.remove(ospj(cls_dir, f))

        prompts = [template.format(dataset_name, classname) for _ in range(n_repeat) for template in templates][:n]
        with open(ospj(cls_dir, "prompts.json"), "w") as f:
            json.dump(prompts, f, indent=4)

        # NHU generate.py: moi lop mot seed, theo vi tri trong danh sach DAY DU, nen chay
        # --classes rieng le ra dung anh nhu chay ca bo.
        set_seed(args.seed * 1000 + all_classes.index(classname))
        t0 = time.time()
        count = 0
        for ks in batch_iteration(list(range(n)), args.bs):
            outputs = pipe(
                prompt=[prompts[k] for k in ks],
                image=[load_ref(ospj(fewshot_dir, classname, refs_log[classname][k])) for k in ks],
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                callback_on_step_end=log_step,
            )
            count = save_data(outputs, cls_dir, count)
        n_new += n
        dt = time.time() - t0
        print("  %-12s %d anh, %.1f s (%.2f s/anh)" % (classname, n, dt, dt / n), flush=True)

    dt = time.time() - t_total
    print("\nXong: %d anh moi, %.1f s (%.2f s/anh)" % (n_new, dt, dt / max(n_new, 1)))
    print("strength=%s, num_inference_steps=%d -> diffusers bo qua %d buoc dau, du kien %d buoc khu nhieu"
          % (args.strength, args.num_inference_steps,
             args.num_inference_steps - int(args.num_inference_steps * args.strength),
             int(args.num_inference_steps * args.strength)))
    if timesteps:
        print("Thuc te: %d lan goi UNet, timestep %s (tren 1000)" % (len(timesteps), timesteps))


if __name__ == "__main__":
    main()
