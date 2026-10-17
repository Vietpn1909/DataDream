## Làm gì

<!-- 1-2 câu. Tiền tố title: [gen] [clf] [eval] [abl] [infra] [docs] -->

## Ảnh hưởng ai

- [ ] Chỉ phần của tôi
- [ ] Sửa code dùng chung (`SUBSET_NAMES`, `configs/`, `scripts/`, `requirements-*`)
      → sau khi merge phải nhắn nhóm `git pull`

## Đã kiểm tra

- [ ] `python scripts/check_env.py` không thêm lỗi mới
- [ ] Nếu sửa chỗ chọn ảnh: `python scripts/test_kshot_consistency.py` pass
- [ ] Nếu đổi phiên bản thư viện bước 1–2: `python scripts/smoke_test_env.py` pass
      và đã sinh lại `requirements-gen.lock.txt`
- [ ] Không commit ảnh / checkpoint / zip / `local.yaml`
- [ ] Nếu đổi `SUBSET_NAMES`: đã sửa **cả 3 file** (`DataDream/util_data.py`,
      `generate/util.py`, `classify/util_data.py`) và chúng giống hệt nhau
- [ ] Nếu thêm số vào `results/*.csv`: đủ cả cột `date` và `commit`

## Người review

<!-- @tên người được phân công review chéo -->
