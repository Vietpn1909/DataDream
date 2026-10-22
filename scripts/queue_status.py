"""Cho hang doi biet bo anh con thieu gi, de tiep tuc thay vi sinh lai tu dau.

    python scripts/queue_status.py <thu muc bo anh> <dataset> <so anh moi lop> [--no-delete]

In ra moi lop CHUA DU mot dong:  <vi tri lop>\t<ten lop>\t<count_start>
count_start = so thu tu anh dau tien con thieu hoac hong (generate.py dat ten
anh 0.png, 1.png, ... nen sinh tiep tu so do). Anh hong (vd. ghi do khi mat dien)
bi xoa, bao ra stderr.
"""
import ast
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
root, dataset, n_img = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
# --no-delete: chi bao, khong xoa - dung khi co tien trinh khac co the dang ghi anh
no_delete = "--no-delete" in sys.argv[4:]

src = (REPO / "generate" / "util.py").read_text(encoding="utf-8")
m = re.search(r"['\"]" + re.escape(dataset) + r"['\"]\s*:\s*\[(.*?)\]", src, re.S)
classes = ast.literal_eval("[" + m.group(1) + "]")


def ok(p):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


for idx, cls in enumerate(classes):
    d = root / "train" / cls
    start = 0
    while start < n_img:
        p = d / ("%d.png" % start)
        if not p.is_file():
            break
        if not ok(p):
            if no_delete:
                print("anh hong (khong xoa vi --no-delete): %s" % p, file=sys.stderr)
            else:
                print("anh hong, da xoa: %s" % p, file=sys.stderr)
                p.unlink()
            break
        start += 1
    if start < n_img:
        print("%d\t%s\t%d" % (idx, cls, start))
