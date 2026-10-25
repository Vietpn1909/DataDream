# Lỗi và bẫy của repo gốc

Mọi mục dưới đây đã được **kiểm trên code hoặc chạy thử thật**, không phải phỏng
đoán. Số dòng ở mục 1–13 tính theo commit gốc `a71515f`. Kiểm nhanh xem các bản sửa
còn nguyên không:

```powershell
python scripts\check_env.py
```

## Tóm tắt

| # | Vấn đề | Trạng thái |
|:-:|---|---|
| 1 | Danh sách lớp phải cắt xuống 20 ở cả 3 module | ✅ đã sửa |
| 2 | Tập test chưa được lọc về 20 lớp | ✅ đã sửa |
| 3 | Classifier chọn ảnh không sort (3 chỗ) | ✅ đã sửa |
| 4 | `assert` chặn `mode=zeroshot` | ✅ đã sửa |
| 5 | `revision="fp16"` khi nạp model local | ✅ đã sửa |
| 6 | Thư viện bản mới làm vỡ code | ✅ đã ghim (lock) |
| 7 | `mixed_precision` mặc định là fp32 | ⚠️ truyền `--mixed_precision=bf16` |
| 8 | Biến thể cls không bao giờ ghi checkpoint | ⚠️ dùng vòng lặp ở `run.md` |
| 9 | `--count_start` không phải cơ chế resume | ⚠️ dùng `--n_set_split` |
| 10 | Sinh ảnh cls giữ 2 pipeline trên GPU → chậm ~10 lần | ✅ đã sửa |
| 11–13 | Chi tiết cần ghi đúng trong báo cáo | 📝 |
| 14 | Nghi resume crash với PyTorch ≥ 2.6 | ✅ đã kiểm: không phải lỗi |
| 15 | Model SD 2.1 gốc đã bị gỡ khỏi Hugging Face | ✅ có `download_sd21.py` |
| 16 | Mặc định `--n_template` không có hiệu lực | ✅ đã sửa |
| 17 | Tập test EuroSAT thực ra là tập train | ✅ đã sửa |
| 18 | Mọi lớp dùng chung seed → ảnh các lớp giống nhau | ✅ đã sửa |
| 19 | Setting "chỉ ảnh thật" không dùng đúng tập few-shot | ✅ đã sửa |
| 20 | Setting "thật + sinh" nhận ảnh thật qua tên thư mục | ✅ đã sửa |

Hiện **không còn lỗi nào đã biết đang mở**. Các lỗi của code đánh giá (mục 2, 17, 19,
20) đã sửa và kiểm bằng `python scripts\test_eval_splits.py`.

---

## Chặn người khác — phải xong trước khi nhóm chạy số

### 1. ✅ Danh sách lớp phải cắt xuống 20 ở cả 3 module

`SUBSET_NAMES['fgvc_aircraft']` trong repo gốc có 100 variant, nằm ở ba file:
`DataDream/util_data.py`, `generate/util.py`, `classify/util_data.py`. Đã cắt cả ba
xuống 20 lớp theo `configs/classes_fgvc_aircraft20.txt`, bằng cùng một khối thay thế
nên ba diff giống hệt nhau.

**Giữ nguyên tên dataset `fgvc_aircraft`**, không tạo tên mới kiểu
`fgvc_aircraft20`. Tên mới sẽ rơi vào nhánh `else: raise ValueError` ở nhiều chỗ so
chuỗi cứng — riêng `classify/` có ít nhất 6 chỗ: `models/clip.py:20`, `data.py:243`,
`:273`, `:553`, `:643`, cộng `SUBSET_NAMES`.

Ba file phải **giống hệt nhau**. Lệch thì nhãn giữa bước sinh ảnh và bước train CLIP
không trùng nhau, và không có gì báo lỗi — chỉ ra accuracy sai. `check_env.py` mục
4 kiểm đúng chuyện này.

20 lớp giữ **đúng thứ tự tương đối** của danh sách 100 lớp, ở các vị trí
`[0, 1, 3, 4, 5, 10, 13, 14, 17, 19, 25, 26, 28, 33, 35, 42, 47, 53, 95, 96]` — cần
cho mục 2.

Đổi danh sách sau khi đã train thì mọi checkpoint và bộ ảnh sinh trước đó hết dùng
được. Phải chốt một lần, sớm, rồi không đổi nữa.

### 2. ✅ Tập test chưa được lọc về 20 lớp

Vấn đề nghiêm trọng nhất, và sẽ lộ ra muộn nhất — lúc TV4 chạy số đầu tiên.

- `classify/data.py:643` nạp tập test bằng
  `tv.datasets.FGVCAircraft(split='test', annotation_level='variant')` → **đủ 100
  variant**, nhãn `0..99` theo thứ tự của torchvision.
- `filter_dset()` (`classify/data.py:234`) chỉ giảm **số ảnh mỗi lớp**, không giảm số
  lớp.
- `classify/main.py:68` lấy `n_classes = len(SUBSET_NAMES[dataset])` = 20.

Model đầu ra 20 chiều nhưng nhận nhãn tới 99 → crash, hoặc tệ hơn là một con số
accuracy vô nghĩa mà không ai nhận ra.

**Cần viết thêm** một bước lọc: giữ các mẫu có nhãn nằm trong 20 vị trí ở mục 1,
rồi đổi nhãn cũ → vị trí trong danh sách 20. Áp cho cả tập test (`data.py:643`) và
nhánh train ảnh thật (`data.py:553`).

Phụ: `filter_dset()` dùng `random.sample` **không đặt seed** (`data.py:263`), nên
baseline ảnh thật không tái lập giữa các lần chạy. Nên nhận một `seed` và truyền
vào.

**Đã sửa:** hàm mới `filter_aircraft_classes()` trong `classify/data.py`, gọi ở cả tập
test và nhánh train ảnh thật. Hàm ánh xạ theo **tên** lớp (không theo danh sách số
viết tay), nên tên nào lệch với torchvision là báo lỗi ngay. `filter_dset()` dùng RNG
riêng seed 0. Kiểm bằng `scripts/test_eval_splits.py`: tập test còn 668 ảnh, 20 nhãn
0..19, 33–34 ảnh/lớp, nhãn mới trỏ đúng tên lớp gốc của cả 668 ảnh. Code cũ cho 3.333
ảnh với nhãn 0..99.

### 3. ✅ Classifier chọn ảnh không sort

Generator chọn K ảnh thật bằng `os.listdir` → `.sort(key=natural_keys)` → lấy K ảnh
đầu (`DataDream/datadream.py:214-217`). Classifier thì có **ba** chỗ dùng
`os.listdir` mà không sort:

| Chỗ | Vai trò |
|---|---|
| `classify/data.py:123` | few-shot thuần |
| `classify/data.py:204` | `is_pooled_fewshot` — ảnh sinh trộn K ảnh thật, tức setting chính của DataDream |
| `classify/data.py:185` | duyệt ảnh sinh rồi cắt theo `n_img_per_cls` |

Hậu quả: classifier train trên K ảnh thật **khác** K ảnh đã fine-tune LoRA, và thứ tự
`os.listdir` khác nhau giữa các máy nên kết quả không tái lập. Chỗ thứ ba làm việc
chọn ảnh sinh nào cũng ngẫu nhiên theo máy, nên ablation "số ảnh mỗi lớp" không tái
lập được.

Đã thêm `atoi`/`natural_keys` vào `classify/utils.py` — bản sao y nguyên của
`DataDream/util.py` — và sort ở cả ba chỗ. Giờ K nhỏ luôn là tập con của K lớn, và
`n_img_per_cls` nhỏ luôn là tập con của `n_img_per_cls` lớn.
`scripts/test_kshot_consistency.py` kiểm điều này.

---

## Chặn mình — phải xong trước khi chạy

### 4. ✅ `assert` chặn `mode=zeroshot`

`generate/generate.py:313` viết `assert mode == "datadream" and n_shot >= 1`, luôn
sai khi `mode=zeroshot` — tức không sinh được bộ ảnh SD gốc. Đã sửa thành phép kéo
theo `assert mode != "datadream" or n_shot >= 1`.

### 5. ✅ `revision="fp16"` khi nạp model local

`generate/generate.py:34` truyền `revision="fp16"` vào
`StableDiffusionPipeline.from_pretrained`. `revision` là tên nhánh trên Hugging Face
Hub, vô nghĩa với thư mục local và làm `from_pretrained` báo lỗi. Đã xoá;
`torch_dtype=torch.float16` ở dòng dưới là đủ.

### 6. ✅ Thư viện bản mới làm vỡ code

Hai import của `DataDream/datadream.py` đã bị xoá khỏi bản mới của thư viện:

| Dòng | Import | Vỡ từ |
|---|---|---|
| `datadream.py:36` | `from diffusers.loaders import LoraLoaderMixin` | diffusers 0.30 |
| `datadream.py:51` | `from huggingface_hub.utils import insecure_hashlib` | huggingface_hub 0.26 |

Cộng `check_min_version("0.26.0.dev0")` ở `datadream.py:67` → diffusers phải nằm
trong khoảng 0.26–0.29.

Chỉ đổi tên lớp sang `StableDiffusionLoraLoaderMixin` là **không đủ**:
`datadream.py:537` còn giả định `lora_state_dict` trả về tuple
`(state_dict, network_alphas)`, và `save_lora_weights` cũng đổi chữ ký ở bản mới. Lỗi
sẽ nổ ở bước lưu — sau khi đã train xong.

Môi trường đã ghim trong `requirements-gen.lock.txt`, dựng bằng
`scripts/setup_env.ps1`. `scripts/smoke_test_env.py` chạy lại đúng chuỗi gắn LoRA →
train → lưu → nạp lại trên GPU. Đừng cài `requirements.txt` của tác giả — xem
[setup.md](setup.md#đừng-cài-theo-cách-thông-thường).

### 7. ⚠️ `mixed_precision` mặc định là fp32

`DataDream/config.py:310` để `default=None`, nên không truyền gì là chạy fp32 —
chậm và tốn VRAM.

Truyền **`--mixed_precision=bf16`**, không phải `fp16`: code hỗ trợ sẵn
(`datadream.py:441`), card Blackwell chạy bf16 nguyên tốc độ, và bf16 không cần loss
scaling nên tránh được NaN khi train cả text encoder. Chỉ dùng `fp16` nếu cần khớp
đúng setup của paper, và khi đó phải theo dõi loss.

Bước sinh ảnh thì `generate.py` luôn nạp pipeline ở fp16. Đã sinh thử thật: ảnh bình
thường, không đen, không NaN.

---

## Lưu ý khi chạy — tránh mất một đêm GPU

### 8. ⚠️ Biến thể cls không bao giờ ghi checkpoint

Mỗi LoRA của cls chỉ chạy 400 bước (16 ảnh ÷ batch 8 = 2 bước × 200 epoch), nên
điều kiện `global_step % 1000 == 0` (`datadream.py:841`) không bao giờ đúng →
không có checkpoint → `--resume_from_checkpoint` vô dụng cho đêm chạy 20 lớp.

Thay vào đó, vòng lặp 20 lớp tự bỏ qua lớp đã có `pytorch_lora_weights.safetensors`
— máy tắt giữa chừng thì chạy lại y nguyên, mất tối đa một lớp. Vòng lặp đã chạy thử
ở [run.md](run.md#1-fine-tune-lora).

Ngược lại, biến thể dset (8.000 bước) mặc định giữ **mọi** checkpoint kèm trạng thái
optimizer, vì `checkpoints_total_limit` mặc định `None` (`config.py:163`). Tham số
chung trong `run.md` đã đặt `--checkpoints_total_limit=1`.

### 9. ⚠️ `--count_start` không phải cơ chế resume

`generate/generate.py:250-251` áp `count_start` cho **mọi lớp** trong lần chạy. Treo
ở lớp thứ 12 rồi chạy lại với `--count_start` sẽ sinh đè lên 11 lớp đã xong. Cách
đúng: chạy lại khối lớp còn thiếu bằng `--n_set_split` / `--split_idx` — xem
[run.md](run.md#chạy-qua-đêm).

Kể cả khi chỉ áp cho một lớp, `--count_start` vẫn không chạy tiếp đúng: seed được đặt
lại ở đầu lớp, nên ảnh thứ `count_start` trở đi là **bản sao y hệt** ảnh 0, 1, 2… của
chính lớp đó (cùng prompt, cùng nhiễu). Hàng đợi vì vậy sinh lại cả lớp dở từ ảnh 0.

### 10. ✅ Sinh ảnh cls nạp lại pipeline từng lớp, giữ 2 bản trên GPU

Với `--is_dataset_wise_model=False`, `generate/generate.py:400-402` gọi lại
`get_pipe()` cho từng lớp. Mặt tốt là không bị cộng dồn adapter LoRA giữa các lớp.

Đã đo: nạp pipeline từ ổ D: mất **2,8 giây**, nên 20 lần chỉ thêm khoảng một phút
trên tổng gần hai tiếng — thời gian nạp không đáng tối ưu.

**Nhưng code gốc không giải phóng pipeline cũ** (pipeline nạp ở đầu `main()` vẫn nằm
trên GPU khi nạp cái mới), nên có hai bản SD cùng lúc. Trên card 16 GB, VRAM lên
15,9 / 16,3 GB, Windows đẩy phần dư sang RAM thường và sinh ảnh chậm khoảng 10 lần:
mỗi lô 10 ảnh mất 2,5–4 phút thay vì 15 giây. Không có lỗi nào báo ra, chỉ chậm. Đo
trên 20 ảnh: **186 giây → 43 giây**, VRAM 15,9 → 14,0 GB, ảnh giống hệt từng pixel.
Đã sửa: giải phóng pipeline cũ (`gc.collect()` + `torch.cuda.empty_cache()`) trước
khi nạp cái mới.

---

## Chi tiết cần ghi đúng trong báo cáo

### 11. `target_modules` của U-Net là 6 module, không phải 4

`DataDream/datadream.py:475`:
`["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]`. Text encoder thì
đúng 4 như paper: `q_proj, k_proj, v_proj, out_proj` (`:485`).

Nhưng UNet của SD 2.1 **không có** `add_k_proj`/`add_v_proj`, nên thực tế LoRA chỉ
gắn vào 4 loại module — đã kiểm bằng cách gắn LoRA đúng config này vào UNet thật:
`to_q`, `to_k`, `to_v`, `to_out.0`, mỗi loại 32 chỗ (32 lớp attention), tổng
**3,32 triệu tham số LoRA** ở UNet với rank 16. Báo cáo nên ghi 4 module, kèm chú
thích code khai báo 6.

### 12. Prompt thực tế có thêm tên miền

Template đầu là `a {}photo of a {}`, chỗ `{}` đầu được điền `"aircraft "`
(`generate/generate.py:116`), nên prompt thật là `"a aircraft photo of a 737-300"`.
Paper chỉ ghi gọn `"a photo of a [CLS]"`.

### 13. Chỉ chạy một seed → bảng số không có thanh sai số

Kế hoạch chỉ dùng `seed0`, trong khi paper dùng nhiều fewshot seed. Nhóm sẽ phải bình
luận về các con số lệch 1–2% mà không biết chúng có nằm trong nhiễu không.

Rẻ nhất: chạy thêm `seed1` trên EuroSAT để có một ước lượng độ nhiễu, và ghi rõ hạn
chế một seed trong phần limitations.

---

## Phát hiện khi dựng môi trường

### 14. ✅ Nghi resume crash với PyTorch ≥ 2.6 — đã kiểm, không phải lỗi

Nghi vấn từ đọc mã nguồn: từ PyTorch 2.6, `torch.load` mặc định
`weights_only=True`, trong khi accelerate 0.29.3 gọi `torch.load` không truyền tham số
đó, mà trạng thái lưu có chứa mảng numpy (`np.random.get_state()`).

Chạy thật với torch 2.11.0+cu128 thì **không crash**: một vòng `save_state` →
`load_state` có đủ thành phần như `datadream.py` — scheduler cosine, DataLoader xáo
trộn, RNG numpy đã dùng, bf16 — nạp lại bình thường, learning rate trước và sau khớp
nhau. PyTorch 2.11 đã cho phép mảng numpy ở chế độ `weights_only`.

`smoke_test_env.py` mục 5 giữ bài kiểm này để bắt lại nếu sau này nâng thư viện. Vẫn
nên thử resume bằng chính `datadream.py` trong lần chạy thử EuroSAT —
[run.md](run.md#0-chạy-thử-trên-eurosat).

### 15. ✅ Model SD 2.1 gốc đã bị gỡ khỏi Hugging Face

`stabilityai/stable-diffusion-2-1-base` trả về 401. Dùng
`python scripts\download_sd21.py`: tải từ mirror cộng đồng
`sd2-community/stable-diffusion-2-1-base` (đã kiểm có `model_index.json` và đủ 6
thành phần diffusers), chỉ lấy bản fp32 `.safetensors` (~5 GB) thay vì cả repo
(~37 GB). Không lấy `.bin`/`.ckpt` vì là pickle.

Đã sinh ảnh thật từ model này: ra máy bay nhận ra được.

### 16. ✅ Mặc định `--n_template` không có hiệu lực

`DataDream/config.py:370` viết `args_n_template = 1` (gạch dưới) thay vì
`args.n_template = 1` — gán vào một biến không ai dùng. Quên truyền `--n_template`
thì:

- `TEMPLATES_SMALL[:None]` lấy **cả 27 template** để train thay vì 1 như paper —
  lệch setup mà không báo gì;
- checkpoint ghi vào `…/tplNone/` trong khi `generate.py` tìm `…/tpl1/`.

Đã sửa thành dấu chấm. Phát hiện khi cho parser thật của repo đọc các lệnh trong
`run.md`.

Cùng loại bẫy, nhưng không phải lỗi: `--learning_rate` mặc định là `5e-4`, khác
`1e-4` của paper. Quên truyền thì checkpoint nằm ở `lr0.0005_…/` và `generate.py`
không tìm thấy. Tham số chung trong `run.md` đã có sẵn.

### 17. ✅ Tập test EuroSAT thực ra là tập train

EuroSAT không có tập test chính thức; `classify/` dùng file chia của CoOp,
`split_zhou_EuroSAT.json`, gồm ba phần `train` (13.500 ảnh), `val` (5.400), `test`
(8.100). Hàm `split_eurosat(file_path, split, dataset)` lọc ảnh theo phần được
truyền vào.

Nhưng **cả nhánh train lẫn nhánh test đều truyền `'train'`**:

```python
# classify/data.py, nhánh train ảnh thật (~dòng 590)
train_dataset = split_eurosat(eurosat_path_train, 'train', train_dataset)
# classify/data.py, get_test_dataset (~dòng 679)
test_dataset = split_eurosat(eurosat_path_test, 'train', test_dataset)
```

Tức là classifier được **chấm điểm trên chính 13.500 ảnh của phần train**. Accuracy
EuroSAT sẽ bị thổi phồng, và mọi so sánh dùng EuroSAT (ablation) đều sai. Sửa: đổi
`'train'` thành `'test'` ở dòng thứ hai.

Đáng ghi vào báo cáo: nếu số liệu EuroSAT của paper được chạy bằng chính code này
thì chúng cũng mắc lỗi tương tự — nên so với số của paper phải thận trọng.

Ngoài ra, nhánh test của cả Aircraft lẫn EuroSAT đọc từ `real_train_data_dir` chứ
không phải `real_test_data_dir`, nên giá trị điền ở key test trong `local.yaml` không
có tác dụng gì.

Ảnh few-shot EuroSAT đã được lấy đúng từ phần `train` của file chia này
(`scripts/make_fewshot_eurosat.py`), nên sau khi sửa dòng trên, few-shot và tập
test không trùng nhau.

**Đã sửa:** đổi thành `'test'`. Kèm theo, `split_eurosat()` so đường dẫn bằng set (code
gốc so ~27.000 × 8.100 lần) và chuẩn hoá `/` – `\` — code gốc chạy trên Windows ra
**0 ảnh** vì lệch dấu gạch. Kiểm bằng `scripts/test_eval_splits.py`: tập test 8.100
ảnh, đủ 10 nhãn, không trùng ảnh nào với phần train, không chứa ảnh few-shot nào.

### 18. ✅ Mọi lớp dùng chung seed → ảnh các lớp giống nhau

`generate.py` gốc gọi `set_seed(seed)` trước *mỗi* lớp, nên ảnh thứ k của mọi lớp
bắt đầu từ cùng một nhiễu. Hậu quả: ảnh thứ k của hai lớp khác nhau có tương quan
pixel 0,76 (SD gốc EuroSAT), 0,79 (SD gốc Aircraft), 0,85 (DataDream EuroSAT). Ở
DataDream EuroSAT, 6/10 lớp còn trùng luôn nội dung.

Đã sửa thành mỗi lớp một seed (`seed * 1000 + vị trí lớp`). Sau khi sửa, độ trùng
về 0. Tăng guidance từ 2 lên 5 làm ảnh giống lớp thật hơn (+10 điểm theo CLIP chưa
train), nhưng paper cố ý dùng 2.0 để ảnh đa dạng hơn. Vì vậy giữ 2.0 cho tới khi bước 3
có accuracy thật.

Ba bộ đã sinh trước khi sửa (SD gốc Aircraft, SD gốc EuroSAT, DataDream EuroSAT)
**chưa bàn giao**, cần sinh lại. Toàn bộ chẩn đoán, số liệu và hình cho báo cáo:
[investigations/class-collapse.md](investigations/class-collapse.md).

### 19. ✅ Setting "chỉ ảnh thật" không dùng đúng tập few-shot

Ba setting của bước 3 lấy ảnh thật từ hai nguồn khác nhau:

- **thật + sinh** (`--is_pooled_fewshot=True`): đọc `real_train_fewshot_data_dir` —
  đúng 16 ảnh/lớp mà generator đã fine-tune.
- **chỉ ảnh thật** (`--is_synth_train=False`): code gốc lấy `n_img_per_cls` ảnh
  **ngẫu nhiên, không seed** từ toàn bộ phần train qua `filter_dset()` — một tập khác.

Tức là baseline "chỉ ảnh thật" và DataDream không dùng cùng ảnh thật, so sánh trong
bảng chính không công bằng (paper so trên cùng tập few-shot). Không báo lỗi gì.

**Đã sửa:** `DatasetFewshotReal` trong `classify/data.py`. Khi `--n_shot > 0`, nhánh
ảnh thật đọc thẳng `real_train_fewshot_data_dir/<lớp>/`, sort `natural_keys`, lấy
`n_shot` ảnh đầu — cùng cách với setting thật + sinh và với `DataDream/datadream.py`.
`--n_shot=0` giữ nhánh gốc (dùng khi muốn baseline nhiều ảnh thật hơn).

Ghi chú: với EuroSAT seed0, sau khi `filter_dset()` được đặt seed 0 (mục 2) thì tình cờ
bốc trúng đúng tập few-shot, vì `make_fewshot_eurosat.py` dùng cùng thuật toán và
seed. Không có gì đảm bảo điều đó cho Aircraft, seed hay K khác — nên vẫn phải sửa.

### 20. ✅ Setting "thật + sinh" nhận ảnh thật bằng chuỗi `real_train` trong đường dẫn

`DatasetSynthImage.__getitem__` gốc đặt `is_real = "real_train" in image_path`. Thư mục
few-shot không có chữ đó — ví dụ tải từ Kaggle (`real/eurosat_fewshot_seed0_K16/`) —
thì **mọi ảnh thật bị coi là ảnh sinh**: `loss_real` tính trên tập rỗng, có thể ra
`nan`, hoặc trọng số `--lambda_1` mất tác dụng. Không báo lỗi gì. Đã kiểm trên code
gốc: 0/20 ảnh thật được nhận là thật.

**Đã sửa:** ghi lại ranh giới (ảnh sinh ở đầu danh sách, ảnh thật nối sau) và dựa vào
vị trí thay vì đường dẫn.

---

## Đã kiểm là đúng

Để không ai mất thời gian xác minh lại:

- **Compute hai biến thể bằng nhau:** dset = 320 ảnh ÷ batch 8 = 40 bước × 200 epoch
  = 8.000 bước; cls = 2 bước × 200 epoch × 20 lớp = 8.000 bước.
- **LoRA:** `rank` mặc định 16 (`config.py:352`), `lora_alpha = rank`,
  `init_lora_weights="gaussian"` (`datadream.py:472-474`).
- **Chọn biến thể:** không truyền `--target_class_idx` → dset; truyền `=i` → cls cho
  lớp thứ `i`.
- **Đường dẫn khớp giữa hai bước:** `datadream.py` ghi checkpoint ra
  `outputs/<dataset>/shot16_seed0_tpl1/lr0.0001_epoch200/<dataset-wise|tên lớp>/`,
  và `generate.py` tìm đúng đường đó. Cả hai cùng thêm `_notextlora` khi tắt LoRA
  text encoder. Đường dẫn của từng biến thể đã lấy từ chính code, liệt kê ở
  [run.md](run.md).
- **Cờ trùng thì cờ sau thắng**, ở cả `datadream.py` (argparse) lẫn `generate.py`
  (Fire) — các biến thể trong `run.md` dựa vào điều này.
- **Sinh ảnh SD gốc không cần ảnh thật:** `mode=zeroshot` chỉ đọc danh sách lớp,
  `model_dir` và `save_dir`. Chạy được ngay khi chốt danh sách lớp, không phải chờ
  dữ liệu.
- **`accelerate launch` chạy được trên Windows** mà không cần
  `accelerate config`: một tiến trình, nhận đúng `bf16`.
- **`local.yaml` mở bằng đường dẫn tương đối** ở cả ba module → phải `cd` vào đúng
  thư mục trước khi chạy.
- **`classify/main.py:11` `import wandb` vô điều kiện** (assert ở `:135` là code chết)
  → phải cài `wandb` dù dùng tensorboard.
