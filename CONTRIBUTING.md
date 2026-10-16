# Quy ước làm việc

Năm người, một repo, một GPU. Mấy quy ước dưới đây tồn tại để không ai phải
chạy lại một đêm GPU vì người khác đổi code.

## 1. `master` luôn chạy được

Nhánh chính của repo này tên là **`master`**, không phải `main`. Không ai push
thẳng vào nó.

**Mỗi người một nhánh duy nhất, đặt đúng bằng tên mình:**

| người | nhánh | phụ trách |
|---|---|---|
| Việt | `viet` | fine-tune LoRA, sinh ảnh |
| TV2 | `tv2` | dữ liệu thật, tập few-shot |
| TV3 | `tv3` | sinh ảnh, FID |
| TV4 | `tv4` | train CLIP |
| TV5 | `tv5` | ablation, ghép báo cáo |

Làm xong một việc thì mở Pull Request từ nhánh của mình vào `master`, **1 người
duyệt**, rồi merge. Sau khi merge vẫn làm tiếp trên cùng nhánh đó — nhánh là của
người, không phải của việc.

Vì nhánh sống lâu, hai điều phải giữ:

- **Merge vào `master` thường xuyên**, đừng dồn nhiều việc vào một PR. Nhánh càng
  lâu không merge thì PR càng to, review càng hình thức, và càng dễ đụng nhau.
- **`git pull origin master`** vào nhánh mình trước khi làm tiếp, mỗi lần có người
  khác merge. Dùng tiền tố commit ở mục 2 để PR nhiều việc vẫn đọc được.

> Đừng đặt tên kiểu `viet/abc`. Git lưu nhánh thành đường dẫn file, nên không thể
> tồn tại đồng thời nhánh `viet` và nhánh `viet/abc` — có cái này là không tạo
> được cái kia, cả ở local lẫn trên GitHub.

## 2. Tiền tố commit message

| tiền tố | phạm vi |
|---|---|
| `[gen]` | fine-tune LoRA, sinh ảnh (`DataDream/`, `generate/`) |
| `[clf]` | classifier (`classify/`) |
| `[eval]` | FID, phân tích, chỉ số ảnh |
| `[abl]` | ablation |
| `[infra]` | cấu hình repo, requirements, script dùng chung |
| `[docs]` | README, báo cáo, slide |

Ví dụ: `[gen] cắt SUBSET_NAMES aircraft xuống 20 lớp`.

## 3. Không commit những thứ này

Đã chặn trong `.gitignore`, nhưng để rõ ràng: **không** commit ảnh (thật hay
sinh), checkpoint `.safetensors`/`.pt`, file zip, log tensorboard, và **ba file
`local.yaml`**. Dùng Drive chung.

Riêng `classify/local.yaml` có key `wandb_key`. Đừng dán API key thật vào bất
cứ file nào trong repo, kể cả notebook.

## 4. Đổi code dùng chung thì phải báo

Danh sách lớp (`SUBSET_NAMES`), `requirements-*.txt`, `configs/` và mọi thứ
trong `scripts/` là code **dùng chung**. Sửa những chỗ đó thì sau khi merge
phải nhắn vào nhóm: *"đã merge X, mọi người `git pull` trước khi chạy tiếp"*.

Đặc biệt với `SUBSET_NAMES`: ai đang có checkpoint LoRA hoặc bộ ảnh sinh trước
khi danh sách lớp đổi thì **những thứ đó hết dùng được** — nhãn lệch hết. Vì
vậy danh sách lớp phải chốt một lần, sớm, rồi không đổi nữa.

## 5. Ghi kết quả vào `results/`

Dùng đúng header cột trong `results/results_template.csv`, kể cả hai cột `date`
và `commit`. Chi tiết ở [results/README.md](results/README.md). Mỗi người một
file, append vào cuối, đừng sắp xếp lại.

## 6. Notebook Colab

Mỗi người một file trong `notebooks/`: `tv4_clip.ipynb`, `tv5_ablation.ipynb`...
Đừng sửa notebook của người khác.

Ô đầu tiên của mọi notebook:

```python
from google.colab import drive, userdata
drive.mount('/content/drive')

import os
REPO = '/content/datadream-reproduce'
token = userdata.get('GITHUB_TOKEN')   # chỉ cần nếu repo private
url = f'https://{token}@github.com/Vietpn1909/DataDream.git' if token else 'https://github.com/Vietpn1909/DataDream.git'
if os.path.exists(REPO):
    !cd {REPO} && git pull
else:
    !git clone {url} {REPO}
%cd {REPO}
```

Nếu repo để private: mỗi người tự tạo một fine-grained token **chỉ có quyền với
repo này**, lưu trong mục Secrets của Colab với tên `GITHUB_TOKEN`. Không dán
token thẳng vào notebook.

Lưu notebook về repo bằng *File > Save a copy in GitHub*. **Xóa output ảnh lớn
trước khi commit** — notebook kèm ảnh làm repo phình rất nhanh và diff vô dụng.

## 7. Trước khi báo "xong"

```bash
python scripts/check_env.py
```

Và với mỗi bộ dữ liệu bàn giao: đếm đủ số ảnh mỗi lớp, xem lưới một ảnh đại
diện mỗi lớp, có `README.txt` kèm theo, tên zip đúng mẫu, giải nén thử ra đúng
cấu trúc `train/<tên lớp>/`. Một bộ ảnh lỗi làm phí cả buổi train CLIP trên Colab.

```bash
# đếm số ảnh mỗi lớp
for d in <thư mục>/train/*/; do echo "$(ls "$d"*.png | wc -l)  $d"; done
```
