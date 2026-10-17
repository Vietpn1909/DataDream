#!/usr/bin/env python3
"""Tao 3 file local.yaml tu configs/local.example.yaml.

    python scripts/setup_local_yaml.py

Khong ghi de file da ton tai, nen chay lai bao nhieu lan cung an toan.
Chay giong nhau tren Windows, Linux va Colab.
"""
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "local.example.yaml"


def main():
    if not SRC.is_file():
        sys.exit("LOI: khong thay " + str(SRC))

    created = 0
    for d in ("DataDream", "generate", "classify"):
        dst = ROOT / d / "local.yaml"
        if dst.exists():
            print("  bo qua  " + d + "/local.yaml (da co)")
        else:
            shutil.copyfile(SRC, dst)
            print("  tao     " + d + "/local.yaml")
            created += 1

    print()
    if created:
        print("Da tao %d file. VIEC TIEP THEO: mo tung file va thay" % created)
        print("/home/USER/... bang duong dan tuyet doi tren may ban.")
        print("Tren Windows dung dau / , vi du: D:/DataDream/data/eurosat")
        print("Doc chu thich trong file - co may key rat de dien sai.")
    else:
        print("Ca 3 file da co san, khong thay doi gi.")
    print("\nKiem tra lai bang:  python scripts/check_env.py")


if __name__ == "__main__":
    main()
