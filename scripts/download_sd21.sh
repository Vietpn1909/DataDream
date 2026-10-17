#!/usr/bin/env bash
# Tải Stable Diffusion 2.1 base (512) về máy, chỉ những file code cần (~5 GB).
#
#   bash scripts/download_sd21.sh                 # tải vào ~/models/sd2.1-base
#   bash scripts/download_sd21.sh /duong/dan/khac
#
# Vì sao không dùng lệnh trong kế hoạch ban đầu:
#   - stabilityai/stable-diffusion-2-1-base đã bị Stability AI gỡ khỏi
#     Hugging Face (API trả về 401). Dùng bản mirror cộng đồng
#     sd2-community/stable-diffusion-2-1-base, cùng cấu trúc diffusers.
#   - Tải nguyên repo mirror là ~37 GB vì có nhiều bản trùng (.ckpt, .bin,
#     fp16, nonema). Code chỉ cần bản fp32 dạng .safetensors: ~5 GB.
#   - Chỉ lấy .safetensors và .json, KHÔNG lấy .bin/.ckpt: hai định dạng đó là
#     pickle, nạp vào có thể chạy code tùy ý - không nên tin từ repo cộng đồng.
#
# Cần huggingface_hub đã cài (có trong requirements-gen.txt).
set -euo pipefail

DEST="${1:-$HOME/models/sd2.1-base}"
REPO="sd2-community/stable-diffusion-2-1-base"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "LỖI: chưa có huggingface-cli. Chạy: pip install -r requirements-gen.txt" >&2
    exit 1
fi

echo "Tải $REPO -> $DEST"
mkdir -p "$DEST"

# --local-dir-use-symlinks False: ghi file thật vào DEST thay vì symlink sang
# ~/.cache, để xoá cache sau này không làm hỏng model.
huggingface-cli download "$REPO" \
    --include "model_index.json" \
              "feature_extractor/*" \
              "scheduler/*" \
              "tokenizer/*" \
              "text_encoder/config.json" "text_encoder/model.safetensors" \
              "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
              "vae/config.json" "vae/diffusion_pytorch_model.safetensors" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False

echo
echo "Kiểm tra:"
missing=0
for f in model_index.json \
         text_encoder/model.safetensors \
         unet/diffusion_pytorch_model.safetensors \
         vae/diffusion_pytorch_model.safetensors \
         tokenizer/vocab.json \
         scheduler/scheduler_config.json; do
    if [ -s "$DEST/$f" ]; then
        printf "  ok     %8s  %s\n" "$(du -h "$DEST/$f" | cut -f1)" "$f"
    else
        printf "  THIẾU            %s\n" "$f"
        missing=1
    fi
done
echo "  tổng: $(du -sh "$DEST" | cut -f1)"

if [ "$missing" -ne 0 ]; then
    echo "Có file thiếu - chạy lại script (huggingface-cli tải tiếp phần còn thiếu)." >&2
    exit 1
fi

echo
echo "Xong. Điền đường dẫn TUYỆT ĐỐI này vào model_dir.sd2.1 trong local.yaml:"
echo "  $(cd "$DEST" && pwd)"
