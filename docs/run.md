# Chạy fine-tune và sinh ảnh

Lệnh cho PowerShell, sau khi đã `.venv\Scripts\activate` (xem [setup.md](setup.md)).
Mọi lệnh ở đây đã được kiểm bằng chính parser của repo: đúng tên cờ, và đường dẫn
đầu ra trong các bảng là đường dẫn code thật sự ghi ra.

- [0. Chạy thử trên EuroSAT](#0-chạy-thử-trên-eurosat) — đo tốc độ, thử resume
- [1. Fine-tune LoRA](#1-fine-tune-lora)
- [2. Sinh ảnh](#2-sinh-ảnh)
- [Chạy qua đêm](#chạy-qua-đêm)

---

## Tham số chung cho fine-tune

Phải `cd` vào `DataDream/` — code đọc `local.yaml` bằng đường dẫn tương đối.

```powershell
cd D:\DataDream\DataDream
$common = @(
    "--fewshot_seed=seed0", "--n_shot=16", "--n_template=1",
    "--train_text_encoder=True", "--train_batch_size=8",
    "--gradient_accumulation_steps=1", "--learning_rate=1e-4",
    "--lr_scheduler=cosine", "--lr_warmup_steps=100", "--num_train_epochs=200",
    "--mixed_precision=bf16", "--checkpointing_steps=1000",
    "--checkpoints_total_limit=1", "--report_to=tensorboard", "--output_dir=outputs"
)
```

- **Đừng bỏ `--learning_rate=1e-4`.** Mặc định của code là `5e-4`; checkpoint sẽ
  nằm ở `lr0.0005_epoch200/` trong khi `generate.py` tìm ở `lr0.0001_epoch200/`.
- **`bf16`, không phải `fp16`**: không cần loss scaling nên không bị NaN khi train cả
  text encoder. Không truyền gì thì code chạy fp32 — chậm và tốn VRAM.
- Cờ đặt **sau** `@common` ghi đè cờ cùng tên bên trong. Các biến thể bên dưới dùng
  cách này.

---

## 0. Chạy thử trên EuroSAT

Mục đích: lấy số đo tốc độ để cả nhóm chốt quy mô, và thử resume trước khi chạy
thật qua đêm. EuroSAT nhẹ và có sẵn trong code.

**Chuẩn bị:** 16 ảnh mỗi lớp vào `D:/DataDream/data/eurosat/real_train_fewshot/seed0/<lớp>/`,
với 10 lớp tên đúng như sau: `AnnualCrop`, `Forest`, `HerbaceousVegetation`,
`Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`,
`SeaLake`.

**Fine-tune 2 epoch** (10 lớp × 16 ảnh ÷ batch 8 = 20 bước/epoch → 40 bước):

```powershell
accelerate launch datadream.py --dataset=eurosat @common --num_train_epochs=2
```

Ghi lại: **giây mỗi bước** (thanh tqdm) và **VRAM tối đa** (`nvidia-smi` ở cửa sổ
khác). Từ đó: một lần fine-tune đầy đủ = 8.000 bước × số giây mỗi bước.

**Thử resume:** chạy lại với `--checkpointing_steps=20`, bấm `Ctrl+C` sau bước 25,
rồi chạy lại y nguyên lệnh đó cộng `--resume_from_checkpoint=latest`. Phải tiếp tục
từ bước 20, không phải từ đầu.

**Sinh thử:**

```powershell
cd ..\generate
python generate.py --sd_version=sd2.1 --mode=zeroshot --dataset=eurosat --n_template=1 `
    --n_img_per_class=10 --guidance_scale=2.0 --bs=10 --n_set_split=1 --split_idx=0
```

**Đã đo (2026-10-17, RTX 5070 Ti):**

| Việc | Tốc độ | VRAM đỉnh |
|---|---|---|
| Sinh ảnh SD gốc, 512×512, 50 bước, `bs=10` | 1,49 giây/ảnh → 4.000 ảnh ≈ 1 giờ 40 phút | 7,4 GB |
| Fine-tune | chưa đo | chưa đo |

---

## 1. Fine-tune LoRA

Cần ảnh 16-shot của Aircraft từ TV2 (xem [data.md](data.md)).

**dset** — một LoRA cho cả 20 lớp, 8.000 bước:

```powershell
accelerate launch datadream.py --dataset=fgvc_aircraft @common
```

**cls** — một LoRA mỗi lớp, 20 lần × 400 bước. Vòng lặp **tự bỏ qua lớp đã xong**,
nên máy tắt giữa chừng thì chạy lại y nguyên:

```powershell
$classes = Get-Content ..\configs\classes_fgvc_aircraft20.txt | Where-Object { $_.Trim() -and -not $_.StartsWith("#") }
for ($i = 0; $i -lt $classes.Count; $i++) {
    $done = "outputs/fgvc_aircraft/shot16_seed0_tpl1/lr0.0001_epoch200/$($classes[$i])/pytorch_lora_weights.safetensors"
    if (Test-Path $done) { "bo qua lop $i ($($classes[$i])) - da xong"; continue }
    accelerate launch datadream.py --dataset=fgvc_aircraft "--target_class_idx=$i" @common
    if ($LASTEXITCODE -ne 0) { break }
}
```

**Biến thể cho ablation** — thêm cờ sau `@common`:

| Biến thể | Thêm |
|---|---|
| chỉ U-Net (không LoRA text encoder) | `--train_text_encoder=False` |
| K = 1 hoặc 4 ảnh thật | `--n_shot=1` hoặc `--n_shot=4` |
| EuroSAT | đổi `--dataset=eurosat` |

K nhỏ tự là tập con của 16-shot: code sort tên file rồi lấy K ảnh đầu.

**Checkpoint nằm ở** `D:/DataDream/DataDream/outputs/fgvc_aircraft/…`:

| Biến thể | Thư mục |
|---|---|
| dset | `shot16_seed0_tpl1/lr0.0001_epoch200/dataset-wise/` |
| cls, lớp thứ i | `shot16_seed0_tpl1/lr0.0001_epoch200/<tên lớp>/` |
| chỉ U-Net | `shot16_seed0_tpl1_notextlora/lr0.0001_epoch200/dataset-wise/` |
| K = 4 | `shot4_seed0_tpl1/lr0.0001_epoch200/dataset-wise/` |

File cần dùng ở mỗi thư mục là `pytorch_lora_weights.safetensors`.

---

## 2. Sinh ảnh

```powershell
cd D:\DataDream\generate
$gen = @(
    "--sd_version=sd2.1", "--dataset=fgvc_aircraft", "--n_template=1",
    "--n_img_per_class=200", "--guidance_scale=2.0", "--bs=10",
    "--n_set_split=1", "--split_idx=0"
)
$dd = @("--mode=datadream", "--n_shot=16", "--fewshot_seed=seed0",
        "--datadream_lr=1e-4", "--datadream_epoch=200")
```

| Bộ ảnh | Lệnh |
|---|---|
| SD gốc | `python generate.py --mode=zeroshot @gen` |
| DataDream dset | `python generate.py @dd --is_dataset_wise_model=True @gen` |
| DataDream cls | `python generate.py @dd --is_dataset_wise_model=False @gen` |
| chỉ U-Net | `python generate.py @dd --is_dataset_wise_model=True --datadream_train_text_encoder=False @gen` |

- **Bộ SD gốc không cần ảnh thật** — chỉ cần danh sách lớp, chạy được ngay, không
  phải chờ TV2.
- **Giữ `--n_set_split=1 --split_idx=0`.** Mặc định là `n_set_split=5`, tức chỉ sinh
  1/5 số lớp.
- `--datadream_lr` và `--datadream_epoch` phải khớp lúc fine-tune, nếu không
  `generate.py` không tìm thấy checkpoint.
- Các bộ ablation (K = 1/4, chỉ U-Net) dùng `--n_img_per_class=100` theo kế hoạch,
  đặt sau `@gen` để ghi đè. Với K = 1/4 thì đổi luôn `--n_shot` cho khớp lúc
  fine-tune.

**Ảnh nằm ở** `D:/DataDream/synthetic/fgvc_aircraft/sd2.1/gs2.0_nis50/…/train/<tên lớp>/<số>.png`:

| Bộ ảnh | Thư mục giữa |
|---|---|
| SD gốc | `shot0_template1/` |
| DataDream dset | `shot16_seed0_template1_lr0.0001_ep200_dswise/` |
| DataDream cls | `shot16_seed0_template1_lr0.0001_ep200/` |
| chỉ U-Net | `shot16_seed0_template1_lr0.0001_ep200_notextlora_dswise/` |

Mỗi thư mục lớp có thêm `prompts.json`. Đóng gói và bàn giao theo [data.md](data.md).

---

## Chạy qua đêm

**Trước khi đi ngủ:** tắt Sleep và tạm hoãn Windows Update — mất điện hay khởi động
lại là mất cả đêm.

**Nếu máy tắt giữa chừng:**

| Đang chạy | Cách tiếp tục |
|---|---|
| fine-tune dset | chạy lại lệnh cũ cộng `--resume_from_checkpoint=latest` (checkpoint mỗi 1.000 bước) |
| fine-tune cls | chạy lại vòng lặp y nguyên — nó bỏ qua các lớp đã xong |
| sinh ảnh | chạy lại đúng khối lớp còn thiếu bằng `--n_set_split` / `--split_idx` |

Với sinh ảnh, **đừng dùng `--count_start`** để tiếp tục cả lần chạy: nó áp cho mọi
lớp, nên sẽ sinh đè lên các lớp đã xong. Ví dụ 20 lớp với `--n_set_split=4`: mỗi
khối 5 lớp, `--split_idx=2` sinh lại lớp 10–14.

`--resume_from_checkpoint` không giúp được biến thể cls: mỗi lớp chỉ 400 bước, không
bao giờ tới mốc 1.000 bước để ghi checkpoint. Vòng lặp bỏ-qua-lớp-đã-xong thay cho nó.

**Sáng hôm sau, đếm ảnh mỗi lớp:**

```powershell
Get-ChildItem <thư mục bộ ảnh>\train -Directory | ForEach-Object { "{0,4}  {1}" -f (Get-ChildItem $_.FullName -Filter *.png).Count, $_.Name }
```

Đủ 20 dòng, mỗi dòng đúng số ảnh đã đặt. Xem thêm danh sách kiểm tra trước khi bàn
giao ở [data.md](data.md).
