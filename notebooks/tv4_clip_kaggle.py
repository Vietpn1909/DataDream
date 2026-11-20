"""TV4 - chay bang chinh tren Kaggle Notebooks (GPU), dung LAI cac o cua tv4_clip.ipynb.

Kaggle khong co Drive/Secrets cua Colab, va dataset vitphmnguyn/datadream gan san o
/kaggle/input (khong phai tai ~7,9 GB). Script nay chi lam phan rieng cua Kaggle (duong
dan, cai thu vien TRUOC khi import torch, tim cho gan dataset) roi exec cac o cua
notebooks/tv4_clip.ipynb (cung commit) - logic chay/doc log/ghi CSV khong bi chep lai.

Day len Kaggle: kernel-metadata.json voi
    "code_file": "tv4_clip_kaggle.py", "kernel_type": "script", "enable_gpu": true,
    "enable_internet": true, "dataset_sources": ["vitphmnguyn/datadream"]
va dat bien moi truong qua dong TV4_DATASET ben duoi (moi dataset mot kernel: mot phien
Kaggle toi da 12 gio). Ket qua: /kaggle/working/tv4/clf_main.csv + log tung lan chay.
"""
import glob
import json
import os
import re
import subprocess
import sys
import types

DATASET = "eurosat"  # TV4_DATASET: "eurosat" hoac "fgvc_aircraft" (kernel tu ghi de dong nay)
BRANCH = "hnguyen"
WORK = "/tmp/dd"                 # o dia tam cua Kaggle: du cho / anh that / output
REPO = f"{WORK}/DataDream"
KEEP = "/kaggle/working/tv4"     # /kaggle/working duoc Kaggle luu lai sau phien
sys.stdout.reconfigure(line_buffering=True)  # log Kaggle dung thu tu voi output cua subprocess


def sh(cmd, check=False):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise SystemExit(f"lenh hong ({r.returncode}): {cmd}")
    return r.returncode


def sh_out(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.splitlines()


# ---------------------------------------------------------------- 0. kiem may
import socket
print("python", sys.version.split()[0], flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo KHONG CO GPU")
try:
    socket.getaddrinfo("github.com", 443)
except OSError:
    raise SystemExit("Kernel KHONG CO INTERNET. Kaggle chi bat Internet/GPU cho tai khoan da xac minh so "
                     "dien thoai (kaggle.com > Settings > Phone verification), va kernel phai bat Internet.")

# ---------------------------------------------------------------- 1. repo + thu vien
os.makedirs(WORK, exist_ok=True)
os.makedirs(KEEP, exist_ok=True)
if not os.path.isdir(REPO):
    sh(f"git clone -q https://github.com/Vietpn1909/DataDream.git {REPO}", check=True)
sh(f"cd {REPO} && git checkout -q {BRANCH} && git log --oneline -1", check=True)
# KHONG dung nguyen requirements-clf.txt: Kaggle chay Python 3.13, ma numpy==1.26.4, Pillow==10.3.0,
# scikit-learn==1.4.2, PyYAML==6.0.1 khong co wheel cho 3.13 (pip se build tu nguon va hong). Giu
# ban Kaggle co san cho cac goi do (+ tensorboard), chi cai goi classify can ma Kaggle khong co.
# wandb de pip chon: main.py chi `import wandb` (chay --log=tensorboard), 0.17 cu cho 3.13.
sh("pip install -q timm==0.9.16 loralib==0.1.2 munch==4.0.0 wandb gdown tensorboard", check=True)
sh("pip install -q git+https://github.com/openai/CLIP.git", check=True)
# clip/clip.py co `from pkg_resources import packaging`, hong voi setuptools moi
if sh("python -c 'import clip'"):
    sh("pip install -q 'setuptools<70'", check=True)
    sh("python -c 'import clip; print(\"clip ok sau khi ha setuptools\")'", check=True)
sh("python -c \"import numpy, PIL, torch, timm, clip; print('numpy', numpy.__version__, '| Pillow', "
   "PIL.__version__, '| torch', torch.__version__, '| timm', timm.__version__, '| cuda', "
   "torch.cuda.is_available())\"", check=True)
if sh("python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'"):
    raise SystemExit("KHONG CO GPU: bat Accelerator GPU cho kernel (tai khoan phai xac minh so dien thoai)")

# ---------------------------------------------------------------- 2. dataset Kaggle da gan
mounts = sorted(glob.glob("/kaggle/input/**/handoff", recursive=True), key=len)
if not mounts:
    raise SystemExit("khong thay /kaggle/input/**/handoff - kernel chua gan dataset vitphmnguyn/datadream")
HANDOFF = mounts[0]
print("dataset:", HANDOFF, flush=True)
# o 7 goi kagglehub.dataset_download(...) + "/handoff": tra ve cho da gan, khong tai
sys.modules["kagglehub"] = types.SimpleNamespace(dataset_download=lambda *_a, **_k: os.path.dirname(HANDOFF))


# ---------------------------------------------------------------- 3. exec o cua notebook
def ipy_to_py(src):
    """Doi magic cua IPython ma notebook dung sang Python thuong.

    IPython thay {bieu_thuc} trong dong !/% bang gia tri -> o day eval thanh f-string;
    repr() de dau nhay trong lenh khong pha chuoi.
    """
    def fs(cmd):
        return f'eval("f" + repr({cmd.strip()!r}), globals())'

    out = []
    for line in src.splitlines():
        ind = line[: len(line) - len(line.lstrip())]
        s = line.strip()
        m = re.match(r"^(\w+)\s*=\s*!(.*)$", s)
        if m:
            out.append(f"{ind}{m.group(1)} = _sh_out({fs(m.group(2))})")
        elif s.startswith("!"):
            out.append(f"{ind}_sh({fs(s[1:])})")
        elif s.startswith("%cd "):
            out.append(f"{ind}os.chdir({fs(s[4:])})")
        else:
            out.append(line)
    return "\n".join(out)


nb = json.load(open(f"{REPO}/notebooks/tv4_clip.ipynb", encoding="utf-8"))
cells = {"".join(c["source"]).split("\n", 1)[0]: "".join(c["source"])
         for c in nb["cells"] if c["cell_type"] == "code"}


def cell(prefix):
    hits = [s for head, s in cells.items() if head.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


ns = {"__name__": "__tv4__", "_sh": sh, "_sh_out": sh_out, "os": os}


def run_cell(prefix, **override):
    print(f"\n########## {prefix}", flush=True)
    exec(compile(ipy_to_py(cell(prefix)), prefix, "exec"), ns)
    ns.update(override)


os.chdir(REPO)
run_cell("# ===== 2. Tham so")
ns.update(
    WORK=WORK, REPO=REPO, BRANCH=BRANCH, DRIVE_OUT=KEEP, TEAM_DRIVE="/nonexistent",
    DATA=f"{WORK}/data", SYN=f"{WORK}/synthetic", OUT=f"{WORK}/outputs",
    FETCH_KAGGLE=True, FETCH_EUROSAT=DATASET == "eurosat", FETCH_AIRCRAFT=DATASET == "fgvc_aircraft",
    CACHE_AIRCRAFT_TAR=False, DATASETS=[DATASET], RESULTS_CSV=f"{KEEP}/clf_main.csv",
)
run_cell("# ===== 4. Kiem GPU")
run_cell("# ===== 7. Anh sinh")
run_cell("# ===== 8. Anh that")
run_cell("# ===== 9. classify/local.yaml")

# o 10 cua notebook chi kiem EuroSAT; o day kiem dung dataset cua kernel
print("\n########## kiem nhanh", flush=True)
sh(f"cd {REPO} && python scripts/test_eval_splits.py --data {ns['DATA']}")
zs = sh(f"cd {REPO} && python scripts/zeroshot_clip.py --dataset {DATASET}")
if zs:
    raise SystemExit("CLIP zero-shot hong -> moi truong chua chay duoc, dung truoc khi train")

run_cell("# ===== 11. Hyperparameter")
run_cell("# ===== 12. Danh sach lan chay")
run_cell("# ===== 13. Ham chay mot lan")

# ---------------------------------------------------------------- 4. chay (khac o 14: dung som)
# Lan chay nhanh truoc, chi-anh-that (100 epoch, cham nhat voi EuroSAT) sau cung. Hai lan hong
# lien tiep (vd OOM, thieu thu vien) thi dung han, khoi dot het quota GPU.
done = ns["done_runs"]()
todo = [r for r in ns["RUNS"] if r["run_name"] not in done and r["data_error"] is None]
todo.sort(key=lambda r: r["variant"] == "real")
print("se chay:", [r["run_name"] for r in todo], flush=True)
print("bo qua (thieu du lieu):", [r["run_name"] for r in ns["RUNS"] if r["data_error"]], flush=True)
failed, streak = [], 0
for r in todo:
    ok = ns["run_one"](r) is not None
    sys.stdout.flush()
    failed += [] if ok else [r["run_name"]]
    streak = 0 if ok else streak + 1
    if streak >= 2:
        print("!!! 2 lan hong lien tiep -> dung.", flush=True)
        break
print("\nXONG. Hong:", failed or "khong co", flush=True)

run_cell("# ===== 15. Bang so voi Table 1")
sh(f"cp {REPO}/notebooks/tv4_clip.ipynb {KEEP}/ 2>/dev/null; ls -la {KEEP}")
