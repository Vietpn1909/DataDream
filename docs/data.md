# Dữ liệu

Repo **không chứa dữ liệu**: ảnh thật, ảnh sinh và checkpoint đều để trên Drive
chung và trên máy từng người. `.gitignore` đã chặn `data/`, `synthetic/`,
`outputs*/`, ảnh, `.safetensors`, `.zip`.

- [Ảnh thật few-shot](#ảnh-thật-few-shot) — quy tắc chung, Aircraft
- [Ảnh thật few-shot: EuroSAT](#ảnh-thật-few-shot-eurosat) — đã tạo bằng script
- [Ảnh sinh](#ảnh-sinh) — `generate.py` tạo ra
- [Drive chung và bàn giao](#drive-chung-và-bàn-giao)

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

**Tập test** không nằm ở đây: `classify/` tự tải FGVC-Aircraft qua torchvision. Lưu
ý tập đó chưa được lọc về 20 lớp —
[known-issues.md](known-issues.md), mục 2.

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

## Drive chung và bàn giao

```
DataDream_shared/
├── real/         ảnh thật và tập few-shot           (TV2 quản lý)
├── synthetic/    các file zip ảnh sinh              (Việt quản lý)
├── lora/         pytorch_lora_weights.safetensors   (Việt)
└── logs/         log tensorboard và README.txt kèm mỗi bộ
```

Link Drive: _(điền vào đây)_

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

- [ ] Đủ số ảnh mỗi lớp — lệnh đếm ở [run.md](run.md#chạy-qua-đêm)
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
