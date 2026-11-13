"""Bang chinh cua nhom (results/clf_main.csv) dat canh Table 1 cua paper DataDream.

    python scripts/make_table_main.py                       # markdown ra man hinh
    python scripts/make_table_main.py --latex --out x.tex   # LaTeX cho bao cao
    python scripts/make_table_main.py --csv /content/drive/MyDrive/DataDream/outputs/clf_main.csv

Doc:
  --csv       results/clf_main.csv (TV4; ghi boi notebooks/tv4_clip.ipynb)
  --zeroshot  results/clf_zeroshot.csv (Vinh; scripts/zeroshot_clip.py)

Cach khop dong: (dataset, variant, pooled), pooled = cot `setting` ket thuc bang "pooled".
Mot o co nhieu dong (chay lai) thi lay dong CUOI file (quy uoc append-only cua
results/README.md) va in canh bao ra stderr.

So cua paper: Table 1, Kim et al., "DataDream: Few-shot Guided Dataset Generation",
ECCV 2024 (arXiv:2407.10910). CLIP ViT-B/16, 16 shot that, 500 anh sinh/lop, mean+-std
3 seed. Nhom: 200 anh sinh/lop, 1 lan chay, few-shot seed0. Aircraft cua nhom la
Aircraft-20 (20 lop, 668 anh test), paper la 100 lop: KHONG so truc tiep cot Aircraft.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = ("eurosat", "fgvc_aircraft")

# Table 1 (arXiv:2407.10910), cot EuSAT va AirC. None = paper khong co dong nay.
PAPER = {
    "zs-classify": ("38.1", "23.8"),
    "zs-reference": (None, None),
    "sdbase": (None, None),
    "issynth": ("71.3±2.9", "34.5±3.4"),
    "disef": ("69.2±2.7", "26.8±2.2"),
    "cls": ("85.4±2.9", "60.3±0.9"),
    "dset": ("80.3±4.1", "71.2±0.1"),
    "real": ("93.5±0.7", "59.3±2.8"),
    "sdbase+R": (None, None),
    "issynth+R": ("93.9±0.1", "64.8±0.8"),
    "disef+R": ("94.0±0.5", "64.3±0.4"),
    "cls+R": ("93.8±0.3", "68.3±0.4"),
    "dset+R": ("93.4±0.0", "72.3±0.2"),
}

# (khoa, nhan, R, S, cach lay so cua nhom)
#   ("main", variant, pooled) -> clf_main.csv ; ("zs", run_name_suffix) -> clf_zeroshot.csv
#   None -> nhom khong chay (DISEF)
ROWS = [
    ("zs-classify", "CLIP zero-shot (prompt của classify)", "", "", ("zs", "zeroshot_classify")),
    ("zs-reference", "CLIP zero-shot (prompt CoOp)", "", "", ("zs", "zeroshot_reference")),
    None,
    ("sdbase", "SD 2.1 gốc", "", "✓", ("main", "sdbase", False)),
    ("issynth", "IsSynth", "", "✓", ("main", "issynth", False)),
    ("disef", "DISEF", "", "✓", None),
    ("cls", "DataDream_cls", "", "✓", ("main", "cls", False)),
    ("dset", "DataDream_dset", "", "✓", ("main", "dset", False)),
    None,
    ("real", "Real-finetune (chỉ ảnh thật)", "✓", "", ("main", "real", False)),
    None,
    ("sdbase+R", "SD 2.1 gốc", "✓", "✓", ("main", "sdbase", True)),
    ("issynth+R", "IsSynth", "✓", "✓", ("main", "issynth", True)),
    ("disef+R", "DISEF", "✓", "✓", None),
    ("cls+R", "DataDream_cls", "✓", "✓", ("main", "cls", True)),
    ("dset+R", "DataDream_dset", "✓", "✓", ("main", "dset", True)),
]

NOT_RUN = "chưa chạy"
NOT_PLANNED = "không chạy"


def read_csv(path):
    if not os.path.exists(path):
        print(f"[canh bao] khong thay {path} - coi nhu chua co dong nao", file=sys.stderr)
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("acc") or "").strip()]


def index_main(rows):
    out, seen = {}, {}
    for r in rows:
        pooled = (r.get("setting") or "").strip().endswith("pooled")
        key = (r["dataset"].strip(), r["variant"].strip(), pooled)
        seen.setdefault(key, []).append(r["run_name"])
        out[key] = r  # dong cuoi thang
    for key, names in seen.items():
        if len(names) > 1:
            print(f"[canh bao] {key}: {len(names)} dong ({', '.join(names)}) - lay dong cuoi",
                  file=sys.stderr)
    return out


def index_zs(rows):
    return {(r["dataset"].strip(), r["run_name"].split("_", 1)[1]): r for r in rows}


def fmt(x):
    return f"{float(x):.1f}"


def ours(src, ds, main, zs):
    if src is None:
        return NOT_PLANNED
    if src[0] == "zs":
        r = zs.get((ds, src[1]))
    else:
        r = main.get((ds, src[1], src[2]))
    return fmt(r["acc"]) if r else NOT_RUN


def delta(o, p):
    try:
        return f"{float(o) - float(p.split('±')[0]):+.1f}"
    except (ValueError, AttributeError):
        return ""


def build(main, zs):
    table = []
    for row in ROWS:
        if row is None:
            table.append(None)
            continue
        key, label, r, s, src = row
        eu, ac = (ours(src, ds, main, zs) for ds in DATASETS)
        p_eu, p_ac = PAPER[key]
        table.append([label, r, s, eu, p_eu or "—", delta(eu, p_eu), ac, p_ac or "—"])
    return table


HEAD = ["Phương pháp", "R", "S", "EuroSAT (nhóm)", "EuroSAT (paper)", "Δ",
        "Aircraft-20 (nhóm)", "Aircraft-100 (paper)¹"]

NOTES = [
    "Paper: Table 1 của DataDream (ECCV 2024), CLIP ViT-B/16, 16 shot thật, 500 ảnh sinh/lớp, "
    "mean±std 3 seed. Nhóm: cùng backbone, 16 shot (seed0), 200 ảnh sinh/lớp, 1 lần chạy; "
    "acc = top-1 ở epoch cuối trên tập test.",
    "R + S: ảnh thật lặp lại cho cân với ảnh sinh, λ = 0,8 (như paper).",
    "¹ Aircraft của nhóm là Aircraft-20 (20 lớp, 668 ảnh test); paper dùng 100 lớp, nên hai cột "
    "Aircraft không so trực tiếp được (bài toán 20 lớp dễ hơn nhiều).",
    "Δ = nhóm − paper (chỉ EuroSAT). DISEF: nhóm không chạy.",
]


def to_markdown(table):
    lines = ["| " + " | ".join(HEAD) + " |", "|" + "|".join(["---"] + [":-:"] * (len(HEAD) - 1)) + "|"]
    for row in table:
        if row is None:
            lines.append("| " + " | ".join([""] * len(HEAD)) + " |")
        else:
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines + [""] + [f"- {n}" for n in NOTES]) + "\n"


def tex_escape(s):
    return (s.replace("±", r"$\pm$").replace("✓", r"\checkmark").replace("—", "--")
            .replace("_", r"\_").replace("Δ", r"$\Delta$").replace("λ", r"$\lambda$")
            .replace("¹", r"$^1$"))


def to_latex(table):
    out = [r"\begin{tabular}{lcc|ccc|cc}", r"\toprule",
           " & ".join(tex_escape(h) for h in HEAD) + r" \\", r"\midrule"]
    for row in table:
        out.append(r"\midrule" if row is None else " & ".join(tex_escape(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    out += ["% " + tex_escape(n) for n in NOTES]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "clf_main.csv"))
    ap.add_argument("--zeroshot", default=os.path.join(ROOT, "results", "clf_zeroshot.csv"))
    ap.add_argument("--latex", action="store_true", help="xuat LaTeX thay vi markdown")
    ap.add_argument("--out", default=None, help="ghi ra file thay vi man hinh")
    args = ap.parse_args()

    table = build(index_main(read_csv(args.csv)), index_zs(read_csv(args.zeroshot)))
    text = to_latex(table) if args.latex else to_markdown(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("da ghi", args.out)
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
