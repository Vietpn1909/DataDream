# DataDream — tái lập trên FGVC-Aircraft

Bản tái lập của nhóm cho paper
**[DataDream: Few-shot Guided Dataset Generation](https://arxiv.org/abs/2407.10910)**
(Kim, Bader và cộng sự, 2024).

Ý tưởng của paper: chỉ với vài ảnh thật mỗi lớp, fine-tune LoRA cho Stable
Diffusion 2.1 để nó sinh ảnh sát với phân phối thật hơn, rồi dùng bộ ảnh sinh đó
để train bộ phân loại CLIP.

Phạm vi của nhóm: **20 lớp của FGVC-Aircraft**, 16 ảnh thật mỗi lớp, cộng EuroSAT
để làm ablation. Code dựa trên
[ExplainableML/DataDream](https://github.com/ExplainableML/DataDream); README gốc
của tác giả được giữ ở [README_upstream.md](README_upstream.md).

## Pipeline

| Bước | Việc | Thư mục | Người | Chạy trên |
|:-:|---|---|---|---|
| 0 | Chuẩn bị ảnh thật và tập 16-shot | — | TV2 | — |
| 1 | Fine-tune LoRA cho SD 2.1 | `DataDream/` | Việt | RTX 5070 Ti, Windows |
| 2 | Sinh ảnh tổng hợp | `generate/` | Việt, TV3 | RTX 5070 Ti, Windows |
| 3 | Train CLIP trên ảnh sinh | `classify/` | TV4, TV5 | Colab |

Bước 1 có hai biến thể, cùng tổng 8.000 bước train nên so sánh được với nhau:
**dset** — một LoRA cho cả dataset, và **cls** — một LoRA riêng cho từng lớp.

## Bắt đầu

**Bước 1–2** — Windows, GPU NVIDIA, đã có [`uv`](https://docs.astral.sh/uv/) và
Python 3.11:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1   # venv + thư viện, ~5 phút lần đầu
.venv\Scripts\activate
python scripts\download_sd21.py D:/models/sd2.1-base             # model, ~5 GB
python scripts\setup_local_yaml.py                                # rồi điền đường dẫn
python scripts\check_env.py
```

Chi tiết và lỗi hay gặp: [docs/setup.md](docs/setup.md). Lệnh fine-tune và sinh
ảnh: [docs/run.md](docs/run.md).

**Bước 3** — Colab: ô đầu tiên của notebook ở
[CONTRIBUTING.md](CONTRIBUTING.md#6-notebook-colab), thư viện ở
[requirements-clf.txt](requirements-clf.txt).

**Dữ liệu** — đặt ảnh ở đâu, đặt tên thế nào, bàn giao qua Drive ra sao:
[docs/data.md](docs/data.md).

## Việc còn mở

> Hai lỗi trong code đánh giá của bước 3 (`classify/data.py`), chưa có người nhận.
> Cả hai không báo lỗi gì, chỉ cho ra accuracy sai — phải sửa trước khi chạy số đầu
> tiên:
>
> - **Tập test Aircraft chưa được lọc về 20 lớp** — code tự tải đủ 100 loại máy bay.
>   [known-issues.md, mục 2](docs/known-issues.md).
> - **Tập test EuroSAT thực ra là tập train** — accuracy EuroSAT bị thổi phồng.
>   [known-issues.md, mục 17](docs/known-issues.md).

## Tài liệu

| File | Đọc khi |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | trước commit đầu tiên — nhánh, PR, quy ước |
| [docs/setup.md](docs/setup.md) | dựng môi trường, gặp lỗi khi cài |
| [docs/run.md](docs/run.md) | chạy fine-tune và sinh ảnh |
| [docs/data.md](docs/data.md) | chuẩn bị, đặt tên, bàn giao dữ liệu |
| [docs/known-issues.md](docs/known-issues.md) | lỗi của repo gốc: đã sửa gì, còn gì |
| [results/README.md](results/README.md) | ghi kết quả vào bảng CSV |

## Cấu trúc

```
DataDream/                  bước 1 — fine-tune LoRA
generate/                   bước 2 — sinh ảnh
classify/                   bước 3 — train CLIP
configs/                    danh sách 20 lớp, mẫu local.yaml
docs/                       tài liệu của nhóm
scripts/                    dựng môi trường, tải model, kiểm tra, hàng đợi GPU
results/                    bảng kết quả CSV, hình cho báo cáo
notebooks/                  notebook Colab, mỗi người một file
requirements-gen.lock.txt   môi trường bước 1–2 đã kiểm, cài qua setup_env.ps1
requirements-gen.txt        nguồn để sinh lock
requirements-clf.txt        thư viện bước 3
requirements.txt            của tác giả — đừng cài
```

Ba thư mục `DataDream/`, `generate/`, `classify/` là code gốc cộng các bản sửa của
nhóm; mỗi chỗ sửa có comment `NHOM SUA` (hoặc `NHOM THEM` nếu là code mới) giải
thích lý do. Tìm hết bằng `git grep -n "NHOM SUA\|NHOM THEM"`.

## 20 lớp

707-320 · 727-200 · 737-300 · 737-400 · 737-500 · 747-100 · 747-400 · 757-200 ·
767-300 · 777-200 · A320 · A321 · A330-300 · A380 · ATR-72 · C-130 · Cessna 172 ·
DC-3 · Spitfire · Tornado

Có các cặp rất giống nhau (737-300/400/500, A320/A321) để thể hiện tính
fine-grained của bài toán. Danh sách đã chốt, nằm ở
[configs/classes_fgvc_aircraft20.txt](configs/classes_fgvc_aircraft20.txt).

## Trích dẫn

```bibtex
@article{kim2024datadream,
  title={DataDream: Few-shot Guided Dataset Generation},
  author={Kim, Jae Myung and Bader, Jessica and Alaniz, Stephan and Schmid, Cordelia and Akata, Zeynep},
  journal={arXiv preprint arXiv:2407.10910},
  year={2024}
}
```
