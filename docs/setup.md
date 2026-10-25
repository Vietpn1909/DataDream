# Dựng môi trường

- [Bước 1–2: Windows](#bước-12-windows) — fine-tune và sinh ảnh, trên GPU của Việt
- [Bước 3: Colab](#bước-3-colab) — train CLIP
- [Lỗi hay gặp](#lỗi-hay-gặp)

---

## Bước 1–2: Windows

Chạy thẳng trên Windows, **không cần WSL**. Đã kiểm code không có gì cần Linux: có
`if __name__ == "__main__"`, `num_workers=0` mặc định, không tách đường dẫn bằng
`/` cứng, không gọi lệnh shell, `accelerate launch` chạy được mà không cần file
config. GPU dưới Windows hay WSL đều đi qua cùng một driver nên tốc độ như nhau.

### Yêu cầu

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- **Python 3.10–3.12**, mặc định dùng 3.11. **Không dùng 3.13+**: numpy 1.26.4,
  tokenizers 0.15.2, safetensors 0.4.3, Pillow 10.3.0 không có bản cho 3.13 trên
  Windows, pip sẽ cố build từ source rồi hỏng.
- GPU NVIDIA. Card RTX 50xx (Blackwell, `sm_120`) cần PyTorch build bằng CUDA 12.8
  trở lên — lock đã ghim `torch 2.11.0+cu128`.

### Một lệnh

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Thêm `-Force` để xoá `.venv` cũ và dựng lại. Lần đầu mất khoảng 5 phút, chủ yếu là
tải PyTorch. Script làm năm việc:

1. Tạo venv ở `.venv/` từ Python 3.11 sẵn có trên máy (tìm qua `py -3.11`, hoặc
   truyền `-Python <đường dẫn>`), cấm `uv` tự tải Python.
2. Cài đúng 45 gói trong `requirements-gen.lock.txt` — lock sinh từ môi trường đã
   kiểm chạy được, không giải phụ thuộc lại từ đầu.
3. Giữ mọi thứ bên trong venv: cache của `uv` đặt trong `.venv/.uv-cache` (cùng ổ
   đĩa nên `uv` dùng hardlink, mỗi file chỉ chiếm chỗ một lần), và
   `sitecustomize.py` trỏ `HF_HOME` vào `.venv/hf-home`. Kết quả: **không cài gì
   vào Python hệ thống, không ghi gì ra `~/.cache/huggingface`**. Xoá `.venv` là
   sạch.
4. `uv pip check` để bắt xung đột phụ thuộc.
5. Chạy `scripts/smoke_test_env.py` trên GPU.

### Kích hoạt venv

```powershell
.venv\Scripts\activate
```

Nếu báo *running scripts is disabled on this system*, chạy một lần lệnh dưới đây.
Nó chỉ cho phép script cục bộ của tài khoản bạn:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Trong VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* →
`.venv\Scripts\python.exe`.

### Tải Stable Diffusion 2.1

```powershell
python scripts\download_sd21.py D:/models/sd2.1-base
```

Khoảng 5 GB; bỏ tham số thì tải vào `~/models/sd2.1-base`.

Repo gốc `stabilityai/stable-diffusion-2-1-base` **đã bị gỡ khỏi Hugging Face**
(trả về 401). Script tải từ mirror cộng đồng
`sd2-community/stable-diffusion-2-1-base` — cùng cấu trúc diffusers — và chỉ lấy
bản fp32 `.safetensors` (~5 GB) thay vì cả repo (~37 GB, toàn bản trùng). Không
lấy `.bin`/`.ckpt` vì đó là pickle, nạp vào có thể chạy code tùy ý.

Phải là bản `-base` (512), không phải bản 768: code fine-tune ở độ phân giải 512.

### Đường dẫn: ba file `local.yaml`

Code đọc đường dẫn từ `local.yaml` trong **từng thư mục** — ba file, không phải một.
Cả ba nằm trong `.gitignore`, mỗi người tự tạo:

```powershell
python scripts\setup_local_yaml.py
```

Script copy `configs/local.example.yaml` ra `DataDream/`, `generate/`,
`classify/` (không ghi đè file đã có). Mở từng file và thay `/home/USER/...` bằng
đường dẫn tuyệt đối; trên Windows viết dấu `/`, ví dụ `D:/DataDream/data/eurosat`.
File mẫu giải thích từng key. Ba chỗ dễ sai:

- `fewshot_data_dir` (bước 1) và `real_train_fewshot_data_dir` (bước 3) là **hai key
  tên khác nhau phải trỏ về cùng một thư mục**. Lệch là classifier train trên tập
  few-shot khác tập đã fine-tune LoRA.
- `datadream_dir` (bước 2) phải trỏ vào `--output_dir` của bước 1.
- Code mở `local.yaml` bằng **đường dẫn tương đối**, nên phải `cd` vào đúng thư mục
  (`DataDream/`, `generate/` hay `classify/`) trước khi chạy.

Ai chỉ làm bước 1–2 thì không cần điền `classify/local.yaml`; `check_env.py` sẽ còn
báo lỗi ở file đó, bỏ qua được.

### Kiểm tra

```powershell
python scripts\check_env.py
python scripts\smoke_test_env.py
python scripts\test_kshot_consistency.py
```

| | `check_env.py` | `smoke_test_env.py` | `test_kshot_consistency.py` |
|---|---|---|---|
| cách kiểm | đọc tĩnh mã nguồn và phiên bản | chạy thật trên GPU | chạy logic chọn ảnh |
| cần torch | không | có | không |
| bắt được | quên áp bản sửa, `local.yaml` thiếu, danh sách lớp lệch giữa 3 module | thư viện cũ vỡ khi ghép với PyTorch mới, thiếu `sm_120` | generator và classifier chọn hai tập K-shot khác nhau |
| thời gian | < 1 giây | ~20 giây | < 1 giây |

`smoke_test_env.py` không cần tải model: nó dựng UNet và text encoder tí hon từ
config rồi đi đúng chuỗi gọi hàm của `datadream.py` — gắn LoRA, train một bước bf16,
lưu, nạp lại, so từng bit — cộng một vòng `save_state`/`load_state` của accelerate.
Chạy lại mỗi khi đổi phiên bản thư viện nào.

### Đừng cài theo cách thông thường

- **Đừng cài `requirements.txt`** (của tác giả). Nó không ghim phiên bản và kéo theo
  `openai`, `langchain`, `google-cloud-aiplatform`, `gspread` — không file `.py` nào
  trong repo import tới — rồi dễ nâng `diffusers`/`huggingface_hub` lên bản làm code
  hỏng.
- **Đừng cài hai bước** (`pip install torch` rồi `pip install -r ...`): bước đầu kéo
  numpy 2.x về, bước sau phải hạ xuống, và dễ bị torch bản CPU từ PyPI đè lên.

Muốn đổi phiên bản thư viện: làm theo hướng dẫn ở đầu `requirements-gen.txt` — sửa
file đó, giải lại một lần cho tất cả, chạy `smoke_test_env.py`, rồi sinh lại lock.

---

## Bước 3: Colab

Trên Colab torch/torchvision đã có sẵn — **đừng cài lại**, sẽ phá môi trường và mất
khoảng 10 phút:

```python
!pip install -q -r requirements-clf.txt
!pip install -q git+https://github.com/openai/CLIP.git
```

Ô đầu tiên của notebook (clone repo, mount Drive) ở
[CONTRIBUTING.md](../CONTRIBUTING.md#6-notebook-colab).

Lưu ý:

- Bộ ghim trong `requirements-clf.txt` **chưa được chạy thử end-to-end**. Ai chạy
  đầu tiên mà phải sửa phiên bản nào thì sửa thẳng vào file đó và mở PR `[clf]`.
- `classify/local.yaml` cần đủ 7 key, kể cả `wandb_key` khi dùng tensorboard — thiếu
  một key là `KeyError` ngay khi khởi động.
- `classify/main.py` `import wandb` vô điều kiện, nên phải cài `wandb` dù không dùng.
- Trước khi chạy baseline "chỉ ảnh thật": [known-issues.md](known-issues.md), mục 19.

---

## Lỗi hay gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `running scripts is disabled on this system` khi activate | PowerShell đang ở chế độ `Restricted` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| pip cố build numpy hoặc tokenizers từ source | đang dùng Python 3.13+ | dựng venv bằng `setup_env.ps1` (mặc định 3.11) |
| `no kernel image is available for execution` | PyTorch không có `sm_120` | `setup_env.ps1 -Force`; đừng tự `pip install torch` |
| `torch.cuda.is_available()` ra `False` sau khi cài thêm gói | gói mới kéo torch bản CPU từ PyPI đè lên | `setup_env.ps1 -Force` |
| VS Code báo đỏ `Cannot find module` | VS Code đang dùng Python hệ thống | chọn interpreter `.venv\Scripts\python.exe` |
| `401` khi tải model | dùng repo `stabilityai/...` đã bị gỡ | `python scripts\download_sd21.py` |
| `setup_env.ps1` báo không tìm thấy Python 3.11 | chưa cài 3.11, hoặc thiếu `py` launcher | truyền `-Python <đường dẫn python.exe 3.10–3.12>` |
| `FileNotFoundError` thư mục lớp khi fine-tune | tên thư mục ảnh lệch tên lớp (hoa thường, dấu cách) | sửa tên thư mục cho khớp [configs/classes_fgvc_aircraft20.txt](../configs/classes_fgvc_aircraft20.txt), đừng sửa danh sách |
| `KeyError` khi đọc `local.yaml` | thiếu key, hoặc chạy sai thư mục | `cd` vào đúng thư mục module; so với `configs/local.example.yaml` |
