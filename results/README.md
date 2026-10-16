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

Ghi thêm dòng thì **append vào cuối file**, đừng sắp xếp lại — git merge dễ hơn nhiều.
