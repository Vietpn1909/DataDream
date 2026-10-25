# Dữ liệu

Repo **không chứa dữ liệu**: ảnh thật, ảnh sinh và checkpoint đều để trên kho
chung (Kaggle) và trên máy từng người. `.gitignore` đã chặn `data/`, `synthetic/`,
`outputs*/`, ảnh, `.safetensors`, `.zip`.

- [Ảnh thật few-shot](#ảnh-thật-few-shot) — quy tắc chung, Aircraft
- [Ảnh thật few-shot: EuroSAT](#ảnh-thật-few-shot-eurosat) — đã tạo bằng script
- [Ảnh sinh](#ảnh-sinh) — `generate.py` tạo ra
- [Bàn giao dữ liệu](#bàn-giao-dữ-liệu) — Kaggle, cách dùng trên Colab

---

## Ảnh thật few-shot

Cấu trúc bắt buộc — code tự nối `seed0/` vào sau `fewshot_data_dir` trong
`local.yaml`:

```
<fewshot_data_dir>/
└── seed0/
    ├── 707-320/
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   └── ... (16 ảnh)
    ├── 727-200/
    └── ... (20 thư mục)
```

Trên máy Việt, `<fewshot_data_dir>` là
`D:/DataDream/data/fgvc_aircraft/real_train_fewshot`.

**Ba điều phải đúng:**

1. **Tên thư mục khớp từng ký tự** với
   [configs/classes_fgvc_aircraft20.txt](../configs/classes_fgvc_aircraft20.txt),
   kể cả hoa thường và dấu cách: `Cessna 172`, không phải `cessna_172`. Sai tên là
   `FileNotFoundError` lúc fine-tune.
2. **Đủ 16 ảnh mỗi lớp.**
3. **Thứ tự tên file là thứ tự được chọn.** Code sort tên file theo kiểu tự nhiên
   (`2.jpg` đứng trước `10.jpg`) rồi lấy K ảnh đầu. Tức là ảnh đứng đầu chính là ảnh
   duy nhất của thí nghiệm 1-shot, và 4 ảnh đầu là tập 4-shot. Generator và
   classifier sort giống hệt nhau, nên hai bên luôn dùng cùng một tập.

**Đừng đổi danh sách lớp sau khi đã train.** Nhãn gắn với vị trí trong danh sách;
đổi danh sách là mọi checkpoint và bộ ảnh sinh trước đó hết dùng được.

**Tập test** không nằm ở đây: `classify/` tự tải FGVC-Aircraft qua torchvision rồi
lọc về 20 lớp ([known-issues.md](known-issues.md), mục 2).

---

## Ảnh thật few-shot: EuroSAT

EuroSAT không có tập test chính thức. `classify/` dùng file chia của CoOp,
`split_zhou_EuroSAT.json` (train 13.500 / val 5.400 / test 8.100 ảnh), nên tập
few-shot **phải lấy từ phần `train` của file đó** — lấy chỗ khác là có nguy cơ ảnh
fine-tune lọt vào tập test.

Đã tạo sẵn trên máy Việt, bằng script trong repo:

```powershell
python scripts\make_fewshot_eurosat.py              # seed0, 16 ảnh/lớp
python scripts\make_fewshot_eurosat.py --seed 1     # thêm một seed nếu cần
```

Cách chọn: mỗi lớp, sort tên file trong phần `train`, rồi lấy ngẫu nhiên 16 ảnh bằng
`random.Random(seed)`. Cùng seed ra cùng tập trên mọi máy. Script tự kiểm không ảnh
nào thuộc `val`/`test`, và ghi `manifest.json` liệt kê ảnh đã chọn.

Cấu trúc trên máy, khớp với chỗ `classify/` sẽ đọc:

```
D:/DataDream/data/eurosat/
├── train/
│   ├── eurosat/2750/<lớp>/<ảnh>.jpg      torchvision.datasets.EuroSAT tải về
│   └── split_zhou_EuroSAT.json            file chia của CoOp
└── real_train_fewshot/seed0/
    ├── <lớp>/  × 10, mỗi lớp 16 ảnh
    └── manifest.json
```

File chia tải từ link trong
[DATASETS.md của CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#eurosat).

**Người phụ trách dữ liệu xem lại tập này.** Nếu muốn chọn ảnh theo cách khác thì
các bộ ảnh DataDream EuroSAT đã sinh phải chạy lại (khoảng 1 giờ 40 phút GPU).

---

## Ảnh sinh

`generate.py` ghi vào `save_dir` trong `generate/local.yaml` theo cấu trúc:

```
<save_dir>/<dataset>/sd2.1/gs2.0_nis50/<cấu hình>/train/<tên lớp>/
    ├── 0.png
    ├── 1.png
    ├── ...
    └── prompts.json
```

`<cấu hình>` ứng với từng bộ ảnh xem ở [run.md](run.md#2-sinh-ảnh). Cấu trúc
`train/<tên lớp>/` đọc thẳng được như một `ImageFolder`; `prompts.json` không phải
ảnh nên bị bỏ qua.

Kích thước: ảnh PNG 512×512 khoảng 450 KB mỗi ảnh (đo trên ảnh SD gốc thật), tức
**~1,8 GB cho một bộ 4.000 ảnh**. Cả 9 bộ trong bảng đặt tên bên dưới (26.000 ảnh)
là khoảng 12 GB.

---

## Bàn giao dữ liệu

Ảnh sinh, LoRA và tập few-shot EuroSAT đã bàn giao trên **Kaggle** (công khai):
**<https://www.kaggle.com/datasets/vitphmnguyn/datadream>**

Kaggle đã tự giải nén các file zip, nên trên Kaggle mỗi bộ là một thư mục:

```
handoff/
├── synthetic/                                     ảnh sinh (Việt)
│   ├── eurosat_sdbase_K0_n200/train/<lớp>/<số>.png      + README.txt
│   ├── eurosat_dset_K16_n200/train/...
│   ├── eurosat_cls_K16_n200/train/...
│   └── aircraft20_sdbase_K0_n200/train/...
├── lora/eurosat_lora_K16_seed0/<dataset-wise|lớp>/pytorch_lora_weights.safetensors
└── real/eurosat_fewshot_seed0_K16/seed0/<lớp>/*.jpg   + manifest.json
```

Mỗi bộ có `README.txt` (cạnh thư mục và bên trong): lệnh đã chạy, commit, thời
gian, kết quả kiểm tra, hạn chế đã biết, và **các cờ bắt buộc ở bước 3**.

### Dùng trên Colab

`classify/` đọc ảnh sinh ở
`<synth_train_data_dir>/<dataset>/sd2.1/gs2.0_nis50/<thư mục giữa>/train/`, khác
cấu trúc trên Kaggle. Ô dưới tải dataset (~4,3 GB) rồi tạo symlink đúng chỗ code
cần, không phải chép ảnh:

```python
import os, kagglehub
root = kagglehub.dataset_download("vitphmnguyn/datadream") + "/handoff"
SYN = "/content/synthetic"          # điền vào classify/local.yaml: synth_train_data_dir
for src, dst in {
    "eurosat_sdbase_K0_n200":    "eurosat/sd2.1/gs2.0_nis50/shot0_template1",
    "eurosat_dset_K16_n200":     "eurosat/sd2.1/gs2.0_nis50/shot16_seed0_template1_lr0.0001_ep200_dswise",
    "eurosat_cls_K16_n200":      "eurosat/sd2.1/gs2.0_nis50/shot16_seed0_template1_lr0.0001_ep200",
    "aircraft20_sdbase_K0_n200": "fgvc_aircraft/sd2.1/gs2.0_nis50/shot0_template1",
}.items():
    os.makedirs(os.path.dirname(f"{SYN}/{dst}"), exist_ok=True)
    if not os.path.exists(f"{SYN}/{dst}"):
        os.symlink(f"{root}/synthetic/{src}", f"{SYN}/{dst}")
# tập few-shot EuroSAT: fewshot_data_dir = f"{root}/real/eurosat_fewshot_seed0_K16"
```

Nếu `kagglehub` đòi đăng nhập: `kagglehub.login()`, hoặc đặt `KAGGLE_USERNAME` và
`KAGGLE_KEY` (tạo ở kaggle.com → Settings → API).

Bộ mới (Aircraft DataDream, IsSynth, ablation) sẽ thêm vào **cùng dataset này** dưới
dạng phiên bản mới, giữ nguyên cách đặt tên dưới đây.

### Đặt tên

Một bộ dữ liệu bàn giao = **một file zip + một `README.txt`**, tên theo mẫu
`<dataset>_<biến thể>_K<số shot>_n<số ảnh mỗi lớp>.zip`:

| Bộ | Tên file |
|---|---|
| SD gốc | `aircraft20_sdbase_K0_n200.zip` |
| IsSynth | `aircraft20_issynth_K16_n200.zip` |
| DataDream dset | `aircraft20_dset_K16_n200.zip` |
| DataDream cls | `aircraft20_cls_K16_n200.zip` |
| K = 1, K = 4 | `aircraft20_dset_K1_n100.zip`, `aircraft20_dset_K4_n100.zip` |
| chỉ U-Net | `aircraft20_dset-unetonly_K16_n100.zip` |
| EuroSAT | `eurosat_sdbase_K0_n200.zip`, `eurosat_dset_K16_n200.zip` |

Bên trong zip giữ nguyên cấu trúc `train/<tên lớp>/<số>.png` mà `generate.py` tạo
ra.

### `README.txt` kèm mỗi zip

- lệnh đầy đủ đã chạy
- commit git lúc chạy (`git rev-parse --short HEAD`)
- ngày chạy
- thời gian fine-tune và thời gian sinh ảnh
- số ảnh thực tế mỗi lớp

Các thời gian này cũng là số liệu cho phần compute trong báo cáo, ghi thêm vào
`results/gen_timing.csv`.

### Kiểm tra trước khi bàn giao

Một bộ ảnh lỗi làm phí cả buổi train CLIP trên Colab.

- [ ] Chạy `python scripts\qa_synth.py <thư mục bộ ảnh> <dataset>` — kiểm **toàn bộ**
      ảnh: cấu trúc thư mục, `prompts.json`, ảnh hỏng / sai kích thước / ảnh cũ sót
      lại, ảnh đen, ảnh trùng, lỗi seed chung, và dùng CLIP xem ảnh có đúng chủ đề.
      Phải ra `DAT`; các dòng `[chu y]` thì xem lưới ảnh nó ghi ra `outputs_logs/qa/`
- [ ] Xem lưới vài ảnh mỗi lớp (mỗi lớp một hàng): không có ảnh đen, ảnh nhiễu, hay
      lớp toàn ảnh sai đối tượng; **các hàng không được giống nhau theo cột** — nếu
      giống là lỗi seed chung ([known-issues #18](known-issues.md))
- [ ] Với EuroSAT: chạy `scripts/diag_eval.py` (đo độ trùng giữa các lớp và độ giống
      ảnh thật) — cách đọc ở [investigations/class-collapse.md](investigations/class-collapse.md)
- [ ] So vài lớp dễ nhầm (737-300/400/500, A320/A321) với ảnh thật
- [ ] Có `README.txt`, tên zip đúng mẫu
- [ ] Giải nén thử ra đúng `train/<tên lớp>/`

Xong thì nhắn nhóm: tên file, số ảnh, và lớp nào trông bất thường.

### Thiếu dung lượng Drive

Chuyển ảnh sang JPG chất lượng 95 trước khi nén — CLIP chỉ dùng ảnh 224×224 nên
không mất gì.
