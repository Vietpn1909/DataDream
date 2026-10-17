#!/usr/bin/env python3
"""Kiem tra moi truong LUC CHAY tren GPU - bo sung cho check_env.py (chi doc tinh).

    python scripts/smoke_test_env.py

Khong can tai model: dung UNet va text encoder TI HON dung tu config, nhung goi
DUNG chuoi ham ma DataDream/datadream.py dung de gan LoRA, train, luu va nap lai.
Muc dich la bat nhung cho thu vien doi cu (diffusers 0.27, peft 0.10,
accelerate 0.29) vo khi ghep voi PyTorch moi - loai loi chi lo ra khi dang train.

Chay lai moi khi doi phien ban thu vien nao trong venv.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
FAILED = []


def check(cond, label, detail=""):
    print(("  [ok]   " if cond else "  [LOI]  ") + label)
    if not cond:
        if detail:
            for line in str(detail).splitlines()[-6:]:
                print("         " + line)
        FAILED.append(label)
    return cond


def head(title):
    print("\n=== " + title + " ===")


def run_py(code, cwd=None, env_extra=None, env_drop=()):
    """Chay doan code trong tien trinh con rieng (tranh dung ten module `util`)."""
    env = dict(os.environ)
    for k in env_drop:
        env.pop(k, None)
    env.update(env_extra or {})
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    return r.returncode, (r.stdout + r.stderr).strip()


# --------------------------------------------------------------------------
def test_gpu():
    head("1. GPU va PyTorch")
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("  torch " + torch.__version__ + " | CUDA build " + str(torch.version.cuda))
    if not check(torch.cuda.is_available(), "torch.cuda.is_available()"):
        return False
    dev = torch.device("cuda")
    print("  GPU: %s | compute capability %s" % (
        torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)))
    check("sm_120" in torch.cuda.get_arch_list(), "co sm_120 trong arch list",
          torch.cuda.get_arch_list())

    for dt in (torch.bfloat16, torch.float16):
        a = torch.randn(256, 256, device=dev, dtype=dt)
        check(torch.isfinite(a @ a).all().item(), "matmul %s tren GPU" % str(dt).split(".")[1])

    q = torch.randn(2, 4, 64, 32, device=dev, dtype=torch.bfloat16)
    o = F.scaled_dot_product_attention(q, q, q)
    check(torch.isfinite(o).all().item(), "scaled_dot_product_attention bf16 (attention cua diffusers)")

    x = torch.randn(1, 4, 64, 64, device=dev, dtype=torch.bfloat16)
    w = torch.randn(8, 4, 3, 3, device=dev, dtype=torch.bfloat16)
    check(torch.isfinite(F.conv2d(x, w, padding=1)).all().item(), "conv2d bf16")

    arr = np.arange(6, dtype=np.float32)
    ok = np.allclose(torch.from_numpy(arr).cuda().cpu().numpy(), arr)
    check(ok, "numpy %s <-> torch qua lai (numpy 1.x voi torch build cho numpy 2)" % np.__version__)
    return True


# --------------------------------------------------------------------------
def test_imports():
    head("2. Import thu vien va API ma code dung")
    items = [
        ("diffusers.loaders.LoraLoaderMixin", "from diffusers.loaders import LoraLoaderMixin"),
        ("huggingface_hub.utils.insecure_hashlib", "from huggingface_hub.utils import insecure_hashlib"),
        ("diffusers.training_utils (cast_training_params...)",
         "from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params"),
        ("diffusers.utils (convert_*_to_peft/diffusers)",
         "from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft"),
        ("peft.LoraConfig + peft.utils", "from peft import LoraConfig; from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict"),
        ("transformers CLIPTextModel/AutoTokenizer", "from transformers import CLIPTextModel, AutoTokenizer, PretrainedConfig"),
        ("accelerate Accelerator", "from accelerate import Accelerator"),
        ("fire, ftfy, yaml, tqdm", "import fire, ftfy, yaml, tqdm"),
    ]
    for label, stmt in items:
        try:
            exec(stmt, {})
            check(True, label)
        except Exception as e:  # noqa: BLE001
            check(False, label, repr(e))

    # tensorboard: protobuf duoc keo len ban rat moi, tensorboard 2.16 sinh ma cho
    # protobuf 4.x -> phai GHI THAT mot event moi biet co vo khong.
    try:
        import google.protobuf
        from torch.utils.tensorboard import SummaryWriter
        with tempfile.TemporaryDirectory() as d:
            w = SummaryWriter(d)
            w.add_scalar("loss", 0.5, 1)
            w.close()
            n = len(list(Path(d).glob("events.out.tfevents.*")))
        check(n == 1, "tensorboard ghi duoc event (protobuf %s)" % google.protobuf.__version__)
    except Exception as e:  # noqa: BLE001
        check(False, "tensorboard SummaryWriter", repr(e))


# --------------------------------------------------------------------------
def test_repo_imports():
    head("3. Import chinh code cua repo")
    rc, out = run_py("import datadream; print('ok')", cwd=ROOT / "DataDream")
    check(rc == 0, "import DataDream/datadream.py (gom check_min_version 0.26)", out)
    rc, out = run_py("import generate; print('ok')", cwd=ROOT / "generate")
    check(rc == 0, "import generate/generate.py", out)


# --------------------------------------------------------------------------
def test_lora_path():
    head("4. Duong LoRA cua datadream.py: gan -> train 1 buoc -> luu -> nap lai")
    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DConditionModel
    from diffusers.loaders import LoraLoaderMixin
    from diffusers.training_utils import _set_state_dict_into_text_encoder
    from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import CLIPTextConfig, CLIPTextModel

    dev, wdt = torch.device("cuda"), torch.bfloat16

    def tiny_models():
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=32, attention_head_dim=8,
        )
        te = CLIPTextModel(CLIPTextConfig(
            vocab_size=1000, hidden_size=32, intermediate_size=37,
            num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=77,
        ))
        for m in (unet, te):
            m.requires_grad_(False)
            m.to(dev, dtype=wdt)
        # dung y config cua datadream.py:479-495
        unet.add_adapter(LoraConfig(
            r=16, lora_alpha=16, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        ))
        te.add_adapter(LoraConfig(
            r=16, lora_alpha=16, init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ))
        return unet, te

    try:
        unet, te = tiny_models()
        n_lora = sum(1 for n, p in unet.named_parameters() if p.requires_grad)
        check(n_lora > 0, "unet.add_adapter: %d tham so LoRA duoc train" % n_lora)
        n_te = sum(1 for n, p in te.named_parameters() if p.requires_grad)
        check(n_te > 0, "text_encoder.add_adapter: %d tham so LoRA duoc train" % n_te)
        base_trainable = [n for n, p in unet.named_parameters() if p.requires_grad and "lora" not in n]
        check(not base_trainable, "trong so goc cua UNet van dong bang", base_trainable[:3])

        ids = torch.randint(0, 1000, (2, 77), device=dev)
        latents = torch.randn(2, 4, 8, 8, device=dev, dtype=wdt)
        noise = torch.randn_like(latents)
        t = torch.randint(0, 1000, (2,), device=dev)
        ehs = te(ids)[0]
        pred = unet(latents + noise, t, ehs).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()
        grads = [p.grad for p in list(unet.parameters()) + list(te.parameters()) if p.requires_grad]
        ok = all(g is not None and torch.isfinite(g).all() for g in grads)
        check(ok and torch.isfinite(loss).item(),
              "forward + backward bf16, loss=%.4f, gradient LoRA huu han" % loss.item())

        with tempfile.TemporaryDirectory() as d:
            # dung y save_model_hook, datadream.py:512-529
            LoraLoaderMixin.save_lora_weights(
                d,
                unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
                text_encoder_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(te)),
            )
            f = Path(d) / "pytorch_lora_weights.safetensors"
            check(f.is_file(), "save_lora_weights -> pytorch_lora_weights.safetensors")

            # dung y load_model_hook, datadream.py:545-569
            out = LoraLoaderMixin.lora_state_dict(d)
            check(isinstance(out, tuple) and len(out) == 2,
                  "lora_state_dict tra ve tuple (state_dict, network_alphas)",
                  type(out))
            sd, _alphas = out
            unet2, te2 = tiny_models()
            usd = {k.replace("unet.", ""): v for k, v in sd.items() if k.startswith("unet.")}
            inc = set_peft_model_state_dict(unet2, convert_unet_state_dict_to_peft(usd), adapter_name="default")
            unexpected = getattr(inc, "unexpected_keys", None) or []
            check(not unexpected, "set_peft_model_state_dict khong co key la", unexpected[:3])
            _set_state_dict_into_text_encoder(sd, prefix="text_encoder.", text_encoder=te2)

            def lora_params(m):
                return {n: p for n, p in m.named_parameters() if "lora" in n}
            same_u = all(torch.equal(a, lora_params(unet2)[n]) for n, a in lora_params(unet).items())
            same_t = all(torch.equal(a, lora_params(te2)[n]) for n, a in lora_params(te).items())
            check(same_u, "trong so LoRA UNet nap lai giong het luc luu")
            check(same_t, "trong so LoRA text encoder nap lai giong het luc luu")
    except Exception as e:  # noqa: BLE001
        import traceback
        check(False, "duong LoRA", traceback.format_exc())


# --------------------------------------------------------------------------
RESUME_CODE = """
import tempfile, torch
from accelerate import Accelerator
acc = Accelerator(mixed_precision="bf16")
m = torch.nn.Linear(4, 4)
opt = torch.optim.AdamW(m.parameters())
m, opt = acc.prepare(m, opt)
acc.backward(m(torch.randn(2, 4, device=acc.device)).sum())
opt.step()
with tempfile.TemporaryDirectory() as d:
    acc.save_state(d)
    acc.load_state(d)
print("RESUME_OK")
"""


def test_resume():
    # accelerate 0.29.3 goi torch.load khong truyen weights_only, ma tu PyTorch
    # 2.6 mac dinh la weights_only=True. Da kiem: voi PyTorch 2.11 van nap duoc
    # trang thai RNG chua mang numpy. Giu bai kiem nay de bat lai neu nang
    # PyTorch/accelerate ma vo. Xem docs/known-issues.md muc 14.
    head("5. Resume checkpoint cua accelerate (save_state -> load_state)")
    rc, out = run_py(RESUME_CODE, env_drop=("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",))
    check(rc == 0 and "RESUME_OK" in out,
          "--resume_from_checkpoint chay duoc voi torch.load weights_only mac dinh", out)


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Python " + sys.version.split()[0] + " tai " + sys.executable)
    if test_gpu():
        test_imports()
        test_repo_imports()
        test_lora_path()
        test_resume()
    print("\n" + "=" * 64)
    print("Thoi gian: %.1f giay" % (time.time() - t0))
    if FAILED:
        print("%d muc LOI:" % len(FAILED))
        for f in FAILED:
            print("  * " + f)
        return 1
    print("Tat ca deu chay duoc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
