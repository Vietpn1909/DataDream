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

### Bước 1 + 2 (fine-tune và sinh ảnh) — chạy thẳng trên Windows

Không cần WSL. Code đã được kiểm: có `if __name__ == "__main__"`, `num_workers=0`
mặc định, không tách đường dẫn bằng `/` cứng, không gọi lệnh shell, xformers tắt
mặc định. Dưới Windows hay WSL thì GPU đều chạy qua cùng một driver Windows, nên
tốc độ như nhau.

**Một lệnh dựng toàn bộ môi trường**, giống hệt bản đã kiểm chạy được:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Cần có [`uv`](https://docs.astral.sh/uv/) và Python 3.11 (script tự tìm qua
`py -3.11`). Thêm `-Force` để xoá `.venv` cũ và dựng lại. Script:

1. Tạo venv ở `.venv/` (đã có trong `.gitignore`) từ Python 3.11 sẵn có, cấm `uv`
   tự tải Python.
2. Cài **đúng 45 gói** trong `requirements-gen.lock.txt` — lock sinh từ venv đã kiểm,
   không giải phụ thuộc lại từ đầu.
3. Đặt cache của `uv` **bên trong venv** (cùng ổ đĩa → hardlink, mỗi file chỉ chiếm
   chỗ một lần), và thêm `sitecustomize.py` trỏ `HF_HOME` vào `.venv\hf-home`.
   Kết quả: **không cài gì vào Python hệ thống, không ghi gì ra
   `~/.cache/huggingface`** — xoá `.venv` là sạch.
4. `uv pip check` để bắt xung đột phụ thuộc.
5. Chạy `scripts/smoke_test_env.py` trên GPU (xem dưới).

**Phải dùng Python 3.10–3.12, không dùng 3.13+.** Bộ thư viện đã ghim (numpy
1.26.4, tokenizers 0.15.2, safetensors 0.4.3, Pillow 10.3.0) không có bản cho
3.13 trên Windows.

**Đừng cài theo kiểu hai bước** (`pip install torch` rồi `pip install -r
requirements-gen.txt`): bước đầu kéo numpy 2.x về, bước sau phải hạ xuống, và dễ bị
torch bản CPU từ PyPI đè lên. Lock giải một lần cho cả 45 gói. Cách sinh lock và
lý do dùng `--index-strategy unsafe-best-match` có ghi trong
`scripts/setup_env.ps1`.

Kích hoạt venv để làm việc:

```powershell
.venv\Scripts\activate
```

Nếu báo *running scripts is disabled on this system*, chạy một lần lệnh dưới đây
rồi thử lại. Nó chỉ cho phép script cục bộ của tài khoản bạn:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Không cần `accelerate config default`: với một GPU, `accelerate launch` tự dùng
một tiến trình, còn mixed precision truyền bằng `--mixed_precision` của script.

Trong VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → chọn
`.venv\Scripts\python.exe`, để hết báo đỏ "Cannot find module".

#### Hai bộ kiểm tra — khác nhau thế nào

| | `check_env.py` | `smoke_test_env.py` |
|---|---|---|
cách kiểm | đọc tĩnh mã nguồn và phiên bản | **chạy thật trên GPU** |
cần torch? | không — chạy được ở mọi nơi | có |
bắt được | quên áp patch, `local.yaml` thiếu, danh sách lớp lệch | thư viện cũ vỡ khi ghép với PyTorch mới, thiếu `sm_120` |
thời gian | < 1 giây | ~20 giây |

`smoke_test_env.py` dựng UNet và text encoder **tí hon từ config** (không cần tải
model) rồi đi đúng chuỗi gọi hàm của `datadream.py`: gắn LoRA → train một bước bf16
→ lưu → nạp lại → so từng bit, cộng thêm một vòng `save_state`/`load_state` của
accelerate. Chạy lại mỗi khi đổi phiên bản thư viện nào.

Trên Windows, vòng lặp 20 lớp của biến thể `cls` viết bằng PowerShell thay vì
bash:

```powershell
foreach ($i in 0..19) {
    accelerate launch datadream.py --target_class_idx=$i <các tham số khác>
    if ($LASTEXITCODE -ne 0) { break }
}
```

Nếu `accelerate launch` lỗi riêng trên Windows, chạy thẳng `python datadream.py
<tham số>` — một GPU nên `Accelerator()` tự chạy chế độ một tiến trình.

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

```powershell
python scripts/download_sd21.py D:/models/sd2.1-base   # ~5 GB; bỏ tham số thì vào ~/models/sd2.1-base
```

**Đừng dùng `stabilityai/stable-diffusion-2-1-base`** — Stability AI đã gỡ repo
đó khỏi Hugging Face (trả về 401). Script tải từ bản mirror cộng đồng
`sd2-community/stable-diffusion-2-1-base`, và chỉ lấy bản fp32 `.safetensors`
(~5 GB) thay vì cả repo (~37 GB, toàn bản trùng). Không lấy `.bin`/`.ckpt` vì
đó là pickle, nạp vào có thể chạy code tùy ý.

Phải là bản `-base` (512), không phải bản 768: code fine-tune ở độ phân giải 512.

### Đường dẫn riêng của từng máy

Code đọc đường dẫn từ `local.yaml` nằm trong **từng thư mục** — ba file, không
phải một. Cả ba đã nằm trong `.gitignore`, nên mỗi người tự tạo:

```powershell
python scripts/setup_local_yaml.py
```

Script copy `configs/local.example.yaml` ra `DataDream/local.yaml`,
`generate/local.yaml`, `classify/local.yaml` (không ghi đè file đã có).
Mở từng file và thay `/home/USER/...` bằng đường dẫn thật — trên Windows viết
dấu `/`, ví dụ `D:/DataDream/data/eurosat`. Phần chú thích trong
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
