#!/usr/bin/env bash
# Tạo 3 file local.yaml từ configs/local.example.yaml.
# Không ghi đè file đã tồn tại, nên chạy lại bao nhiêu lần cũng an toàn.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/configs/local.example.yaml"

if [ ! -f "$SRC" ]; then
    echo "LỖI: không thấy $SRC" >&2
    exit 1
fi

created=0
for d in DataDream generate classify; do
    dst="$ROOT/$d/local.yaml"
    if [ -e "$dst" ]; then
        echo "  bỏ qua  $d/local.yaml (đã có)"
    else
        cp "$SRC" "$dst"
        echo "  tạo     $d/local.yaml"
        created=$((created + 1))
    fi
done

echo
if [ "$created" -gt 0 ]; then
    echo "Đã tạo $created file. VIỆC TIẾP THEO: mở từng file và thay"
    echo "/home/USER/... bằng đường dẫn tuyệt đối trên máy bạn."
    echo "Đọc chú thích trong file - có mấy key rất dễ điền sai."
else
    echo "Cả 3 file đã có sẵn, không thay đổi gì."
fi
echo
echo "Kiểm tra lại bằng:  python scripts/check_env.py"
