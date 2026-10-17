# Những chỗ repo gốc chưa chạy được

Tất cả các mục dưới đây đã được **kiểm chứng trực tiếp trên code** trong repo này,
không phải phỏng đoán. Số dòng tính theo commit gốc `a71515f`.

Kiểm tra nhanh xem patch nào đã áp:

```bash
python scripts/check_env.py
```

## Trạng thái

| # | Vấn đề | Trạng thái |
|---|---|---|
| 1 | Cắt danh sách lớp xuống 20 ở 3 module | ✅ đã sửa |
| 2 | **Tập test chưa có bộ lọc lớp** | ❌ **CÒN — chặn bước 3** |
| 3 | K-shot của classifier không sort (3 chỗ) | ✅ đã sửa |
| 4 | `assert` chặn `mode=zeroshot` | ✅ đã sửa |
| 5 | `revision="fp16"` với model local | ✅ đã sửa |
| 6 | Ghim phiên bản thư viện | ✅ có `requirements-gen.txt` |
| 7 | Nên dùng `bf16` thay `fp16` | ⚠️ chọn khi chạy, không phải sửa code |
| 8–10 | Resume, `count_start`, reload pipeline | ⚠️ cần lưu ý khi chạy |
| 11–13 | Chi tiết cho báo cáo | 📝 |
| 14 | Nghi `--resume_from_checkpoint` crash với PyTorch ≥ 2.6 | ✅ đã kiểm: không phải lỗi |
| 15 | Model `stabilityai/stable-diffusion-2-1-base` đã bị gỡ | ✅ có `scripts/download_sd21.py` |

**Mục 2 là việc duy nhất còn chặn người khác.** Nó cần code mới và nên có TV4 xác
nhận, nên tách PR riêng.

---

## Mức CHẶN — phải xong trước khi nhóm chạy số

### 1. ✅ ĐÃ SỬA — cắt danh sách lớp xuống 20 ở cả 3 module

`SUBSET_NAMES['fgvc_aircraft']` trong repo gốc có 100 variant, ở ba file:

| file | vị trí |
|---|---|
| `DataDream/util_data.py` | ~dòng 231 |
| `generate/util.py` | ~dòng 250 |
| `classify/util_data.py` | ~dòng 246 |

**Cách đã làm: giữ nguyên tên `fgvc_aircraft`, cắt list xuống 20 dòng** theo
`configs/classes_fgvc_aircraft20.txt`, dùng cùng một khối thay thế ở cả ba file nên
ba diff giống hệt nhau.

> **Đừng tạo tên dataset mới kiểu `fgvc_aircraft20`.** Nghe thì gọn hơn nhưng
> tên mới sẽ rơi vào nhánh `else: raise ValueError` ở rất nhiều chỗ so chuỗi
> cứng. Riêng `classify/` có ít nhất 6 chỗ phải sửa thêm:
> `classify/models/clip.py:20` (từ điển tên miền thứ ba, dễ bị bỏ sót),
> `classify/data.py:243` và `:273` (nhánh transform),
> `classify/data.py:553` và `:643` (nhánh loader), cộng `SUBSET_NAMES`.
> Giữ nguyên tên thì **không phải sửa một từ điển hay một nhánh `if` nào** ở cả
> ba module.

Ba file phải **giống hệt nhau**. Nếu lệch, nhãn giữa bước sinh ảnh và bước train
CLIP sẽ không trùng nhau và không có gì báo lỗi — chỉ ra accuracy sai.
`scripts/check_env.py` mục 4 kiểm tra đúng chuyện này.

20 lớp đã chọn giữ **đúng thứ tự tương đối** của danh sách 100 lớp (vị trí
`[0, 1, 3, 4, 5, 10, 13, 14, 17, 19, 25, 26, 28, 33, 35, 42, 47, 53, 95, 96]`),
nên nhãn mới = vị trí trong danh sách 20. Cần cho mục 2.

**Hệ quả quan trọng:** sau khi danh sách lớp đổi, mọi checkpoint LoRA và bộ ảnh
sinh ra **trước đó đều hết dùng được**. Vì vậy phải chốt danh sách một lần, sớm,
rồi không đổi nữa.

---

### 2. ❌ CÒN TỒN — tập test chưa có bộ lọc lớp, chưa ai nhận việc này

Đây là vấn đề nghiêm trọng nhất và sẽ lộ ra muộn nhất (khoảng tuần 3, lúc TV4
chạy số đầu tiên).

- `classify/data.py:643` nạp tập test bằng
  `tv.datasets.FGVCAircraft(split='test', annotation_level='variant')` →
  **đủ 100 variant**, nhãn `0..99` theo thứ tự của torchvision.
- `filter_dset()` (`classify/data.py:234`) chỉ giảm **số ảnh mỗi lớp**, hoàn toàn
  không giảm số lớp.
- `classify/main.py:68` lấy `n_classes = len(SUBSET_NAMES[dataset])` = 20.

Kết quả: model có đầu ra 20 chiều nhưng nhận nhãn tới 99 → crash, hoặc tệ hơn là
ra một con số accuracy vô nghĩa mà không ai nhận ra.

**Phải viết thêm** một bước lọc tập test: giữ lại các mẫu có nhãn nằm trong 20 vị
trí ở mục 1, rồi remap nhãn cũ → vị trí trong danh sách 20. Áp cho cả
`get_test_dataset` và nhánh train ảnh thật (`classify/data.py:553`).

**Chưa ai được phân công việc này.** Nó không nằm trong "4 chỗ phải sửa" của kế
hoạch ban đầu, và nó là điều kiện để bước 3 chạy được.

Phụ: `filter_dset()` dùng `random.sample` **không set seed**
(`classify/data.py:263`), nên baseline ảnh thật không tái lập giữa các lần chạy.
Nên nhận một `seed` và truyền vào.

---

### 3. ✅ ĐÃ SỬA — K-shot của classifier không sort (hóa ra là **ba** chỗ)

Lúc viết mục này tôi tưởng chỉ có một chỗ. Khi sửa thì grep ra ba, và chỗ quan
trọng nhất lại không phải chỗ đầu tiên:

| nơi | vai trò | trước khi sửa |
|---|---|---|
| `DataDream/datadream.py:214-217` | generator chọn K-shot | đã `.sort(key=natural_keys)` ✓ |
| `classify/data.py:123` | few-shot thuần | `os.listdir` **không sort** |
| `classify/data.py:204` | **`is_pooled_fewshot`** = ảnh sinh + K ảnh thật | `os.listdir` **không sort** |
| `classify/data.py:185` | duyệt **ảnh sinh** rồi cắt theo `n_img_per_cls` | `os.listdir` **không sort** |

Chỗ `classify/data.py:204` mới là setting chính của DataDream (ảnh sinh trộn với
K ảnh thật), nên nó quan trọng hơn chỗ đầu.

Chỗ `classify/data.py:185` là một ca cùng loại mà kế hoạch ban đầu không nhắc:
vòng lặp này cắt theo `n_img_per_cls`, nên **việc chọn ảnh sinh nào** phụ thuộc
thứ tự `os.listdir` → mỗi máy ra một tập khác, và **ablation "số ảnh mỗi lớp" của
paper không tái lập được**. Sau khi sort thì `n_img_per_cls` nhỏ luôn là tập con
của `n_img_per_cls` lớn, giống đúng cách generator lấy K-shot lồng nhau.

Cách sửa: thêm `atoi`/`natural_keys` vào `classify/utils.py` (bản sao y nguyên của
`DataDream/util.py:18-30` — phải giống hệt, không thì hai bên lại lệch), rồi sort
ở cả ba chỗ.

Mặt tốt: phía generator có sort nên khẳng định "tập K-shot lồng nhau tự động" là
**đúng** — cùng một thư mục 16 ảnh thì `--n_shot=1` và `--n_shot=4` tự là tập con
của 16-shot.

---

## Mức CHẶN mình — sửa trước khi chạy

### 4. ✅ ĐÃ SỬA — `generate.py` lỗi dòng `assert`, không sinh được ảnh SD gốc

`generate/generate.py:313`:

```python
assert mode == "datadream" and n_shot >= 1, \
       "`n_shot` should be integer when `mode` is datadream."
```

Luôn fail khi `mode=zeroshot`. Đã sửa thành:

```python
assert mode != "datadream" or n_shot >= 1, \
       "`n_shot` should be integer when `mode` is datadream."
```

### 5. ✅ ĐÃ SỬA — `generate.py` truyền `revision="fp16"` cho model local

`generate/generate.py:34` truyền `revision="fp16"` vào
`StableDiffusionPipeline.from_pretrained`. Với thư mục local thì `revision`
không có nghĩa và sẽ lỗi. Đã xóa hẳn dòng đó (`torch_dtype=torch.float16` ở dòng
dưới đã đủ).

### 6. Ghim phiên bản thư viện — bắt buộc, không phải cho đẹp

Hai import của `DataDream/datadream.py` đã bị xóa khỏi bản mới của thư viện:

| dòng | import | vỡ từ bản |
|---|---|---|
| `datadream.py:36` | `from diffusers.loaders import LoraLoaderMixin` | diffusers 0.30 |
| `datadream.py:51` | `from huggingface_hub.utils import insecure_hashlib` | huggingface_hub 0.26 |

Cộng với `check_min_version("0.26.0.dev0")` ở `datadream.py:67` → diffusers phải
nằm trong khoảng **0.26–0.29**.

> **Chỉ đổi tên lớp sang `StableDiffusionLoraLoaderMixin` là KHÔNG đủ.**
> `datadream.py:537` còn giả định `lora_state_dict` trả về tuple
> `(state_dict, network_alphas)`, và `save_lora_weights` (`:988`) cũng đổi chữ ký
> ở bản mới. Lỗi sẽ nổ ở bước save/resume — tức là **sau khi đã train xong 4 tiếng**.

Dùng `requirements-gen.txt` (đã ghim) thay vì `requirements.txt` của tác giả.
`requirements.txt` không ghim gì cả và kéo theo `openai`, `langchain`,
`google-cloud-aiplatform`, `gspread`, `oauth2client` — **không file `.py` nào trong
repo import tới**, nhưng pip sẽ phải giải xung đột và rất dễ nâng diffusers lên bản
làm code hỏng.

### 7. `mixed_precision` mặc định là `None` — và nên dùng `bf16`, không phải `fp16`

`DataDream/config.py:310` để `default=None`, nên không truyền cờ là chạy fp32:
chậm hơn và tốn VRAM hơn nhiều.

Nhưng nên truyền **`--mixed_precision=bf16`**, không phải `fp16`:

- Code đã hỗ trợ sẵn (`datadream.py:441`).
- Blackwell chạy bf16 nguyên tốc độ.
- bf16 không cần loss scaling → loại hẳn nguyên nhân NaN / "ảnh sinh ra đen toàn
  bộ" khi fine-tune cả text encoder, đúng sự cố mà kế hoạch đã dự báo.

Chỉ dùng `fp16` nếu cần đối chiếu đúng nguyên setup của paper, và khi đó phải
theo dõi loss xem có NaN.

---

## Mức nên sửa — tránh mất một đêm GPU

### 8. `--checkpointing_steps=1000` vô dụng cho biến thể `cls`

Mỗi LoRA của `cls` chỉ chạy **400 bước** (16 ảnh / batch 8 = 2 bước × 200 epoch),
nên điều kiện `global_step % 1000 == 0` ở `datadream.py:841` **không bao giờ đúng**
→ không có checkpoint nào được ghi → `--resume_from_checkpoint=latest` không dùng
được cho đêm chạy 20 lớp, tức đêm nặng nhất.

May là vòng `for` đã cho hạt resume tự nhiên (mất tối đa một lớp ≈ 13 phút). Chỉ
cần bỏ qua lớp đã xong:

```bash
for i in $(seq 0 19); do
  CKPT=$(ls outputs/fgvc_aircraft/shot16_seed0_tpl1/lr0.0001_epoch200/*/pytorch_lora_weights.safetensors 2>/dev/null | wc -l)
  # hoac kiem tra truc tiep theo ten lop thu i
  accelerate launch datadream.py ... --target_class_idx=$i || break
done
```

Ngược lại, run `dset` (8.000 bước) lại ghi **8 checkpoint full optimizer state** vì
`checkpoints_total_limit` mặc định `None` (`DataDream/config.py:163`). Thêm
`--checkpoints_total_limit=1`.

### 9. `--count_start` không phải cơ chế resume cho bước sinh ảnh

`generate/generate.py:250-251` áp `count_start` cho **mọi lớp** trong lần chạy:

```python
count = self.count_start
prompts = prompts[self.count_start:]
```

Nên nếu treo ở lớp thứ 12, chạy lại với `--count_start` sẽ sinh đè lên 11 lớp đã
xong. Cách đúng: dùng `--n_set_split` / `--split_idx` để chạy lại đúng khối lớp
còn thiếu, và chỉ dùng `count_start` cho một lớp bị dở.

### 10. Sinh ảnh biến thể `cls` nạp lại toàn bộ pipeline 20 lần

`generate/generate.py:400-402`: với `--is_dataset_wise_model=False`, mỗi lớp gọi
lại `get_pipe()` từ đầu. Mặt tốt là không bị cộng dồn adapter LoRA; mặt dở là thêm
~20 × 30–60 giây và dễ phân mảnh VRAM dần. Nên chạy theo `--n_set_split` thành vài
khối thay vì một lần 20 lớp.

---

### 14. ✅ KHÔNG PHẢI LỖI — đã kiểm thực tế, resume chạy được với PyTorch 2.11

**Nghi vấn ban đầu (từ đọc mã nguồn):** từ PyTorch 2.6, `torch.load` mặc định
`weights_only=True`, trong khi accelerate 0.29.3 gọi `torch.load(...)` không truyền
`weights_only` khi nạp trạng thái (`accelerate/checkpointing.py`), mà file
`random_states_0.pkl` chứa `np.random.get_state()` — mảng numpy. Tưởng sẽ bị
từ chối và resume crash.

**Kiểm thật thì không crash.** Trong venv (torch 2.11.0+cu128, accelerate
0.29.3), một vòng `save_state` → `load_state` có đủ thành phần như
`datadream.py` — scheduler cosine của diffusers, DataLoader xáo trộn, RNG numpy
đã dùng, bf16 — nạp lại bình thường, LR trước và sau khớp nhau. PyTorch 2.11 đã
cho phép mảng numpy trong chế độ `weights_only`.

Bản vá `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` từng được thêm vào `datadream.py` đã
**gỡ ra** — để lại một dòng code với comment mô tả lỗi không tồn tại thì tệ hơn
không có.

`scripts/smoke_test_env.py` mục 5 giữ bài kiểm này, để nếu sau này nâng
PyTorch/accelerate mà vỡ thì biết ngay. Vẫn nên thử resume bằng chính
`datadream.py` trong lần chạy thử EuroSAT: train vài trăm bước với
`--checkpointing_steps=100`, dừng giữa chừng, chạy lại với
`--resume_from_checkpoint=latest`.

### 15. ✅ ĐÃ SỬA — model SD 2.1 gốc đã bị gỡ khỏi Hugging Face

`stabilityai/stable-diffusion-2-1-base` trả về 401. Lệnh `huggingface-cli
download` trong kế hoạch sẽ hỏng.

Dùng `python scripts/download_sd21.py`: tải từ mirror cộng đồng
`sd2-community/stable-diffusion-2-1-base` (đã kiểm có `model_index.json` và đủ 6
thành phần diffusers), chỉ lấy bản fp32 `.safetensors` + `.json` (~5 GB) thay vì
cả repo (~37 GB). Không lấy `.bin`/`.ckpt` vì là pickle — nạp vào có thể chạy
code tùy ý, không nên tin từ repo cộng đồng.

---

## Chi tiết cần sửa trong báo cáo

### 11. `target_modules` của U-Net là 6 module, không phải 4

`DataDream/datadream.py:475`:

```python
target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]
```

Có thêm `add_k_proj` và `add_v_proj`. Text encoder thì đúng 4 như paper
(`q_proj, k_proj, v_proj, out_proj`, dòng `:485`).

### 12. Prompt thực tế có thêm tên miền

Template đầu là `a {}photo of a {}` và chỗ `{}` đầu được điền `"aircraft "`
(`generate/generate.py:116`), nên prompt thật là
`"a aircraft photo of a 737-300"` — paper chỉ ghi gọn `"a photo of a [CLS]"`.

### 13. Chỉ chạy một seed → bảng số không có thanh sai số

Kế hoạch chỉ dùng `seed0` cho mọi biến thể, trong khi paper dùng nhiều fewshot
seed. Nhóm sẽ phải bình luận về các con số lệch 1–2% mà không biết chúng có nằm
trong nhiễu hay không.

Rẻ nhất: chạy thêm `seed1` trên **EuroSAT** (nhẹ, đã có sẵn trong code) để có một
ước lượng độ nhiễu, rồi ghi rõ hạn chế một-seed trong phần limitations.

---

## Những khẳng định đã kiểm tra là ĐÚNG

Để không ai mất thời gian xác minh lại:

- Compute hai biến thể bằng nhau: `dset` = 320 ảnh / batch 8 = 40 bước × 200 epoch
  = **8.000 bước**; `cls` = 2 bước × 200 × 20 lớp = **8.000 bước**.
- `rank` mặc định là 16 (`DataDream/config.py:352`), `lora_alpha = rank`,
  `init_lora_weights="gaussian"` (`datadream.py:472-474`).
- Không truyền `--target_class_idx` → train `dset`; truyền `=i` → train riêng lớp
  thứ `i`.
- Đường dẫn checkpoint **khớp chính xác** giữa hai bước: `datadream.py` ghi ra
  `outputs/<dataset>/shot16_seed0_tpl1/lr0.0001_epoch200/<dataset-wise|tên lớp>/`
  (`DataDream/config.py:372-390`) và `generate.py` đi tìm đúng đường đó
  (`generate/generate.py:95-104`). Cả hai cùng thêm hậu tố `_notextlora` khi tắt
  LoRA text encoder.
- Ảnh sinh lưu ở
  `<save_dir>/<dataset>/sd2.1/gs2.0_nis50/<cấu hình>/train/<tên lớp>/<số>.png`.
- Dữ liệu few-shot phải nằm ở `<fewshot_data_dir>/seed0/<tên lớp>/` — code tự nối
  `seed0/` vào (`DataDream/config.py:392-395`).
- **Sinh ảnh SD gốc (`mode=zeroshot`) không cần dữ liệu thật.** `generate.py:315-321`
  chỉ đọc `SUBSET_NAMES`, `model_dir` và `save_dir`. Nghĩa là bộ SD gốc chạy được
  ngay khi chốt xong danh sách 20 lớp, **không phải chờ TV2** — tháo được mấy ngày
  ra khỏi đường găng.
- `classify/config.py:84` mở `"local.yaml"` bằng đường dẫn **tương đối** → bắt buộc
  `cd classify` trước khi chạy. Hai bước kia cũng vậy.
- `classify/main.py:11` `import wandb` là **vô điều kiện** (assert ở `:135` là code
  chết), nên vẫn phải cài `wandb` dù chạy `--log=tensorboard`.
