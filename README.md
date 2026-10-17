# DataDream — bản tái lập của nhóm

Tái lập [DataDream: Few-shot Guided Dataset Generation](https://arxiv.org/pdf/2407.10910)
trên một tập con 20 lớp của FGVC-Aircraft, cộng EuroSAT để làm ablation.

Code gốc: [ExplainableML/DataDream](https://github.com/ExplainableML/DataDream).
README của tác giả được giữ nguyên ở [README_upstream.md](README_upstream.md).

> **Đọc [CONTRIBUTING.md](CONTRIBUTING.md) trước khi commit lần đầu.**
> Repo chỉ chứa code, cấu hình và bảng số liệu nhỏ. Dữ liệu, ảnh sinh và
> checkpoint để trên Drive chung — xem [Dữ liệu dùng chung](#dữ-liệu-dùng-chung).

---

## Ai làm gì

| Bước | Thư mục | Người | Đầu ra |
|---|---|---|---|
| 1. Fine-tune LoRA cho SD 2.1 | `DataDream/` | Việt | checkpoint LoRA (dset, cls) |
| 2. Sinh ảnh | `generate/` | Việt, TV3 | các bộ ảnh tổng hợp |
| 3. Train CLIP | `classify/` | TV4, TV5 | bảng accuracy, bảng ablation |
| Dữ liệu thật + tập few-shot | — | TV2 | `real_train_fewshot/seed0/...` |

Bước 1 và 2 chạy trên RTX 5070 Ti của Việt (16GB, kiến trúc Blackwell / `sm_120`).
Bước 3 chạy trên Colab.

---

## Cài đặt

### Bước 1 + 2 (fine-tune và sinh ảnh)

Trên Windows thì làm **toàn bộ trong WSL2 Ubuntu**, vì repo dùng `accelerate`
và script bash.

```bash
conda create -n datadream python=3.10 -y
conda activate datadream
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gen.txt
```

Index `cu128` là bắt buộc với card RTX 50xx. Kiểm tra ngay:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

Danh sách kiến trúc **phải có `sm_120`**. Không có là bản torch chưa hỗ trợ card,
mọi bước sau sẽ lỗi `no kernel image is available`.

**Đừng cài `requirements.txt`** (của tác giả). File đó không ghim phiên bản và
kéo theo cả `openai`, `langchain`, `google-cloud-aiplatform`, `gspread` — không
file `.py` nào trong repo import tới, nhưng pip sẽ phải giải xung đột phiên bản
và rất dễ nâng `diffusers`/`huggingface_hub` lên bản làm code hỏng. Lý do ghim
từng phiên bản được ghi ngay trong `requirements-gen.txt`.

### Bước 3 (train CLIP)

```bash
pip install -r requirements-clf.txt
pip install git+https://github.com/openai/CLIP.git
```

Trên Colab **đừng** cài lại torch — xem đầu `requirements-clf.txt`.

### Tải Stable Diffusion 2.1 (bản 512)

```bash
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir ~/models/sd2.1-base
```

Phải là bản `-base` (512), không phải bản 768: code fine-tune ở độ phân giải 512.

### Đường dẫn riêng của từng máy

Code đọc đường dẫn từ `local.yaml` nằm trong **từng thư mục** — ba file, không
phải một. Cả ba đã nằm trong `.gitignore`, nên mỗi người tự tạo:

```bash
bash scripts/setup_local_yaml.sh
```

Script copy `configs/local.example.yaml` ra `DataDream/local.yaml`,
`generate/local.yaml`, `classify/local.yaml` (không ghi đè file đã có).
Mở từng file và thay `/home/USER/...` bằng đường dẫn thật. Phần chú thích trong
file mẫu giải thích từng key, kể cả mấy chỗ dễ sai:

- `fewshot_data_dir` (bước 1) và `real_train_fewshot_data_dir` (bước 3) là **hai
  key tên khác nhau phải trỏ về cùng một thư mục**. Lệch là classifier train
  trên tập few-shot khác tập đã fine-tune LoRA.
- `datadream_dir` (bước 2) phải trỏ vào `--output_dir` của bước 1.
- `classify/config.py` mở `local.yaml` bằng **đường dẫn tương đối**, nên bắt
  buộc `cd classify` trước khi chạy `main.py`. Bước 1 và 2 cũng vậy.

### Kiểm tra môi trường

```bash
python scripts/check_env.py
python scripts/test_kshot_consistency.py
```

Chạy cái này trước khi báo "máy em xong rồi" — nó kiểm tra `sm_120`, phiên bản
các thư viện nhạy cảm, sự có mặt của 3 file `local.yaml`, và việc danh sách lớp
trong code có khớp `configs/classes_fgvc_aircraft20.txt` ở cả 3 module hay không.

---

## Cấu trúc repo

```
.
├── DataDream/      # code gốc: fine-tune LoRA            (Việt)
├── generate/       # code gốc: sinh ảnh                  (Việt, TV3)
├── classify/       # code gốc: train CLIP                (TV4, TV5)
├── configs/        # local.example.yaml, danh sách 20 lớp
├── notebooks/      # notebook Colab, mỗi người một file
├── scripts/        # script của nhóm (setup, FID, IsSynth, ablation)
├── results/        # bảng CSV + hình cho báo cáo  (xem results/README.md)
├── requirements-gen.txt    # ghim phiên bản cho bước 1-2
├── requirements-clf.txt    # ghim phiên bản cho bước 3
└── requirements.txt        # của tác giả, GIỮ LÀM THAM KHẢO, đừng cài
```

## 20 lớp dùng trong bài

Danh sách chốt ở [configs/classes_fgvc_aircraft20.txt](configs/classes_fgvc_aircraft20.txt).
Cả 20 tên đã được kiểm tra là có mặt nguyên văn trong `SUBSET_NAMES['fgvc_aircraft']`
của code gốc, và giữ đúng thứ tự tương đối của danh sách 100 lớp (quan trọng cho
việc remap nhãn tập test).

Tên thư mục ảnh thật phải khớp **từng ký tự** với tên lớp, kể cả hoa thường và
dấu cách: `Cessna 172`, không phải `cessna_172`.

---

## Trạng thái các chỗ repo gốc chưa chạy được

Danh sách đầy đủ 13 mục ở [docs/known-issues.md](docs/known-issues.md) — đọc
trước khi chạy bất cứ thứ gì.

**Đã sửa:** cắt danh sách lớp xuống 20 ở cả 3 module; sort khi chọn K-shot và
khi chọn ảnh sinh trong `classify/` (3 chỗ); dòng `assert` chặn `mode=zeroshot`;
`revision="fp16"` khi nạp model local. Cộng `requirements-gen.txt` đã ghim
phiên bản.

**Còn tồn, và nó chặn bước 3:**

> **Tập test chưa có bộ lọc lớp.** `classify/data.py:643` nạp tập test bằng
> `torchvision.datasets.FGVCAircraft` → đủ **100 variant**, nhãn `0..99`, trong
> khi `classify/main.py:68` lấy `n_classes = 20`. `filter_dset()` chỉ giảm số
> ảnh mỗi lớp, không giảm số lớp. Phải viết thêm bộ lọc + remap nhãn, nếu không
> bước 3 sẽ crash hoặc ra accuracy vô nghĩa mà không ai nhận ra.
>
> 20 lớp nằm ở các vị trí `[0, 1, 3, 4, 5, 10, 13, 14, 17, 19, 25, 26, 28, 33,
> 35, 42, 47, 53, 95, 96]` trong danh sách 100 variant, và giữ đúng thứ tự
> tương đối — nên nhãn mới = vị trí trong mảng đó. Chi tiết ở
> [docs/known-issues.md](docs/known-issues.md) mục 2.

**Cần lưu ý khi chạy, không phải sửa code:** dùng `--mixed_precision=bf16` chứ
không phải `fp16`; `--checkpointing_steps=1000` vô dụng với biến thể `cls`;
`--count_start` không phải cơ chế resume. Xem mục 7–10.

---

## Dữ liệu dùng chung

Mọi thứ nặng để trên Drive chung, **không commit**:

```
DataDream_shared/
├── real/         # dữ liệu thật và tập few-shot        (TV2 quản lý)
├── synthetic/    # các file zip ảnh sinh               (Việt quản lý)
├── lora/         # pytorch_lora_weights.safetensors    (Việt)
└── logs/         # log tensorboard và README.txt kèm mỗi bộ
```

Link Drive: _(điền vào đây)_

Quy ước đặt tên zip: `<dataset>_<biến thể>_K<shot>_n<ảnh mỗi lớp>.zip`,
ví dụ `aircraft20_dset_K16_n200.zip`. Bên trong giữ nguyên cấu trúc
`train/<tên lớp>/<số>.png` mà `generate.py` tạo ra, để đọc thẳng như một
`ImageFolder`. Mỗi zip kèm một `README.txt` ghi: lệnh đầy đủ đã chạy, commit
git, ngày chạy, thời gian fine-tune, thời gian sinh ảnh, số ảnh thực tế mỗi lớp.

Ảnh PNG 512×512 khá nặng (~1,6 GB cho một bộ 4.000 ảnh). Nếu Drive miễn phí
không đủ, chuyển sang JPG chất lượng 95 trước khi nén — CLIP chỉ dùng ảnh 224.

## Trích dẫn

Nếu dùng lại code, trích dẫn paper gốc — xem [README_upstream.md](README_upstream.md).
