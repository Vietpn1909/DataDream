# Dung moi truong Python cho buoc 1-2 (fine-tune + sinh anh) tren Windows,
# GIONG HET moi truong da kiem chay duoc (requirements-gen.lock.txt).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Force   # xoa .venv cu, dung lai
#
# Nguyen tac: KHONG cai gi vao Python he thong, KHONG ghi gi ra ngoai venv.
#   - Moi goi cai vao <repo>\.venv (uv chi cho cai vao venv).
#   - Cache cua uv dat TRONG venv (cung o dia -> uv dung hardlink, file chi
#     chiem cho mot lan). Xoa .venv la sach.
#   - sitecustomize.py trong venv tro HF_HOME vao <venv>\hf-home, de
#     transformers/diffusers/accelerate khong ghi vao ~/.cache/huggingface.
#
# Yeu cau: uv, va Python 3.10-3.12 (mac dinh lay 3.11 qua `py -3.11`).
# Python 3.13+ KHONG dung duoc: numpy 1.26.4, tokenizers 0.15.2, safetensors
# 0.4.3, Pillow 10.3.0 khong co ban cho 3.13 tren Windows.
#
# File nay chi dung ky tu ASCII: PowerShell 5.1 doc .ps1 khong co BOM theo bang
# ma ANSI, co dau tieng Viet la vo.

param(
    [string]$Python = "",
    [string]$VenvDir = "",
    [string]$UvCacheDir = "",
    [switch]$Force,
    [switch]$SkipSmokeTest
)

$Root = Split-Path -Parent $PSScriptRoot
if (-not $VenvDir) { $VenvDir = Join-Path $Root ".venv" }
if (-not $UvCacheDir) { $UvCacheDir = Join-Path $VenvDir ".uv-cache" }
$Lock = Join-Path $Root "requirements-gen.lock.txt"
$TorchIndex = "https://download.pytorch.org/whl/cu128"

function Fail([string]$msg) {
    Write-Output ""
    Write-Output "LOI: $msg"
    exit 1
}

function Step([string]$msg) {
    Write-Output ""
    Write-Output "=== $msg ==="
}

# --------------------------------------------------------------------- 0
Step "0. Kiem tra cong cu"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "chua co uv. Cai theo https://docs.astral.sh/uv/getting-started/installation/"
}
Write-Output "uv: $(uv --version)"
if (-not (Test-Path $Lock)) { Fail "khong thay $Lock" }

if (-not $Python) {
    $Python = (py -3.11 -c "import sys; print(sys.executable)" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $Python) {
        Fail "khong tim thay Python 3.11 (py -3.11). Truyen -Python <duong dan python.exe 3.10-3.12>."
    }
    $Python = $Python.Trim()
}
$ver = (& $Python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if (@("3.10", "3.11", "3.12") -notcontains $ver) {
    Fail "Python $ver tai $Python - can 3.10, 3.11 hoac 3.12."
}
Write-Output "Python goc: $ver tai $Python"

# --------------------------------------------------------------------- 1
Step "1. Tao venv tai $VenvDir"
if (Test-Path $VenvDir) {
    if (-not $Force) {
        Fail "$VenvDir da ton tai. Them -Force de xoa va dung lai."
    }
    Write-Output "Xoa venv cu (-Force)..."
    Remove-Item -Recurse -Force $VenvDir -Confirm:$false
}
# --no-cache: luc nay chua co venv de dat cache vao.
uv venv $VenvDir --python $Python --no-python-downloads --no-cache
if ($LASTEXITCODE -ne 0) { Fail "uv venv that bai" }
$VPy = Join-Path $VenvDir "Scripts\python.exe"

# --------------------------------------------------------------------- 2
Step "2. Cai dung cac goi trong lock"
$env:UV_CACHE_DIR = $UvCacheDir
Write-Output "cache cua uv: $UvCacheDir"
# unsafe-best-match: index cua PyTorch cung chua ban sao numpy/tqdm..., nhung
# khong du ban. Chien luoc mac dinh cua uv chi lay mot goi tu index DAU TIEN co
# goi do -> khong giai duoc. best-match la cach pip --extra-index-url van chay.
# torch/torchvision van bi khoa bang hau to +cu128 nen khong the lan sang ban
# CPU tren PyPI.
uv pip sync --python $VPy $Lock --extra-index-url $TorchIndex --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { Fail "uv pip sync that bai" }

# --------------------------------------------------------------------- 3
Step "3. Giu moi thu cua Hugging Face ben trong venv"
$sc = @'
# Chi ton tai trong venv cua du an DataDream (xoa .venv la mat theo).
# Tao boi scripts/setup_env.ps1.
#
# Dua moi thu Hugging Face ghi ra (cache, file danh dau phien ban, lock khi
# tai, config cua accelerate) vao BEN TRONG venv, thay vi vao
# ~/.cache/huggingface toan cuc cua may. Python tu chay file nay moi lan khoi
# dong venv, ke ca khi goi thang .venv\Scripts\python.exe hay qua
# `accelerate launch`.
#
# setdefault: neu ban da tu dat HF_HOME thi gia tri cua ban duoc giu nguyen.
import os
import sys

os.environ.setdefault("HF_HOME", os.path.join(sys.prefix, "hf-home"))
'@
$scPath = Join-Path $VenvDir "Lib\site-packages\sitecustomize.py"
[System.IO.File]::WriteAllText($scPath, $sc, (New-Object System.Text.UTF8Encoding $false))
$hf = (& $VPy -c "import os; print(os.environ['HF_HOME'])").Trim()
Write-Output "HF_HOME trong venv: $hf"

# --------------------------------------------------------------------- 4
Step "4. Kiem tra rang buoc giua cac goi"
uv pip check --python $VPy
if ($LASTEXITCODE -ne 0) { Fail "uv pip check bao xung dot" }

# --------------------------------------------------------------------- 5
if (-not $SkipSmokeTest) {
    Step "5. Chay thu tren GPU"
    & $VPy (Join-Path $Root "scripts\smoke_test_env.py")
    if ($LASTEXITCODE -ne 0) { Fail "smoke_test_env.py co muc loi" }
}

Write-Output ""
Write-Output "Xong. Kich hoat venv:  $VenvDir\Scripts\activate"
Write-Output "Neu bi chan 'running scripts is disabled':"
Write-Output "  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
exit 0
