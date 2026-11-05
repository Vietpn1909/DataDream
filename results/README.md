# results/

Chỉ chứa **bảng số liệu nhỏ (CSV)** và **hình dùng cho báo cáo** (`results/figures/`).
Không để ảnh sinh, checkpoint hay log tensorboard ở đây — những thứ đó lên Drive chung.

## Quy ước cột (bắt buộc giống nhau giữa mọi người)

Mọi file `results/*.csv` dùng đúng header trong `results_template.csv`:

| cột | ý nghĩa | ví dụ |
|---|---|---|
| `run_name` | tên lần chạy, trùng với tên zip đã bàn giao | `aircraft20_dset_K16_n200` |
| `dataset` | tên dataset trong code | `fgvc_aircraft` |
| `variant` | biến thể dữ liệu train | `real`, `sdbase`, `issynth`, `dset`, `cls`, `dset-unetonly` |
| `K` | số shot thật dùng để fine-tune (0 nếu không dùng) | `16` |
| `n_img` | số ảnh sinh mỗi lớp | `200` |
| `seed` | fewshot seed | `0` |
| `setting` | cấu hình classifier | `clip-lora`, `clip-linear` |
| `acc` | top-1 accuracy trên tập test (%) | `62.4` |
| `date` | ngày chạy, `YYYY-MM-DD` | `2026-11-05` |
| `commit` | 7 ký tự đầu commit git lúc chạy | `a71515f` |
| `note` | ghi chú ngắn, để trống nếu không có | `OOM, bs=5` |

Hai cột `date` và `commit` là bắt buộc: không có chúng thì một tháng sau
không ai dựng lại được con số trong bảng của báo cáo.

## Mỗi người một file, đừng sửa chung một file

- `results/gen_timing.csv` — Việt: thời gian fine-tune và sinh ảnh (số liệu cho phần compute)
- `results/clf_main.csv` — TV4: bảng chính
- `results/clf_ablation.csv` — TV5: bảng ablation
- `results/fid.csv` — TV3: FID và các chỉ số ảnh
- `results/diag_class_collapse.csv` — Việt: chẩn đoán lỗi sập lớp, cột riêng do `scripts/diag_eval.py` ghi ([giải thích](../docs/investigations/class-collapse.md))
- `results/clf_zeroshot.csv` — Vinh: CLIP ViT-B/16 zero-shot (chưa train, tức điểm xuất phát của
  classifier) trên đúng tập test của `classify/`, ghi bằng `scripts/zeroshot_clip.py --csv`. Hai dòng
  mỗi dataset: prompt `classify` (đầu phân loại classify bắt đầu train) và `reference` (template
  zero-shot của CoOp); khoảng tin cậy Wilson 95% nằm ở cột `note`. Aircraft ở đây là **Aircraft-20**:
  không so với 23,8% của paper (100 lớp)

Ghi thêm dòng thì **append vào cuối file**, đừng sắp xếp lại — git merge dễ hơn nhiều.
