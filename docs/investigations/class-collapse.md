# Ảnh sinh của các lớp giống nhau ("sập lớp")

Ghi lại lỗi, cách chẩn đoán và cách giải quyết. Viết để dùng lại khi làm báo cáo,
mọi con số đều tái lập được bằng lệnh ở [cuối trang](#tái-lập).

| | |
|---|---|
| Phát hiện | 2026-10-19, sau khi sinh xong bộ DataDream EuroSAT (dset, 2.000 ảnh) |
| Chẩn đoán | 2026-10-20, commit `ef4c71f` |
| Trạng thái | Nguyên nhân 1 **đã sửa** trong `generate/generate.py`. Guidance giữ 2.0 như paper, chờ accuracy của bước 3 rồi mới quyết. Còn 3 lớp chưa giải quyết |
| Số liệu | [results/diag_class_collapse.csv](../../results/diag_class_collapse.csv) |
| Hình | `results/figures/diag_eurosat_*.jpg` |

## Tóm tắt

Bộ DataDream EuroSAT đầu tiên có **6/10 lớp ra gần như cùng một cảnh**. Ảnh thứ k
của lớp này gần như là bản sao ảnh thứ k của lớp khác, và nội dung không giống ảnh
thật của lớp. Có hai nguyên nhân độc lập:

1. **Code gốc dùng chung một seed cho mọi lớp.** Ảnh thứ k của mọi lớp bắt đầu từ
   cùng một nhiễu nên có chung bố cục. Bộ SD gốc cũng bị, không riêng DataDream.
   → Đã sửa: mỗi lớp một seed. Độ giống nhau giữa các lớp giảm từ **0,85 về 0,00**.
2. **Guidance 2.0 cho ảnh bám lớp yếu.** Tăng lên 5.0 thì tỉ lệ ảnh giống đúng lớp
   thật tăng từ **43,5% lên 53–55%** (đo bằng CLIP chưa train). Nhưng paper cố ý
   dùng guidance thấp để ảnh đa dạng hơn, mà chỉ số này không đo độ đa dạng. Vì vậy
   **chưa kết luận được guidance 5 tốt hơn cho classifier**. Giữ 2.0 như paper cho tới
   khi bước 3 có số, xem [So với paper](#so-với-paper).

Sau cả hai bước, **Forest, SeaLake, River vẫn sai** ở mọi cấu hình, kể cả SD gốc.
Vấn đề này nằm ngoài seed và guidance, xem [mục Còn mở](#còn-mở).

## Phát hiện thế nào

Không có lỗi nào được báo: fine-tune xong, sinh đủ 2.000 ảnh, file nào cũng mở được.
Lỗi chỉ lộ ra khi xếp ảnh của các lớp thành lưới, mỗi lớp một hàng
([hình 1](#hình)). Nhìn theo cột thì thấy Forest, HerbaceousVegetation, Pasture,
PermanentCrop, River và một phần Highway có cùng bố cục, chỉ khác nhau chút màu.

Đó là lý do bộ này **không được bàn giao** cho bước 3: train CLIP trên ảnh mà một
cảnh mang 6 nhãn khác nhau thì kết quả DataDream sẽ sai lệch, mà không ai biết vì
sao.

## Đã loại trừ

Trước khi đi tìm nguyên nhân, tôi kiểm xem LoRA có được nạp đúng không:

- **LoRA được nạp đủ:** sau `pipe.load_lora_weights`, text encoder có 3,01M tham số
  LoRA, U-Net có 3,32M, khớp với file checkpoint (440 tensor).
- **LoRA có tác dụng lên text encoder:** embedding prompt thay đổi. Cosine trung bình
  giữa prompt của hai lớp khác nhau tăng từ 0,940 lên 0,949, tức là LoRA làm các lớp
  *gần nhau hơn* một chút. Đây là gợi ý đầu tiên cho nguyên nhân 2.
- **Prompt lúc train và lúc sinh giống nhau:** cả hai đều là
  `"a satellite photo of a <tên lớp>"` (template 0 của `TEMPLATES_SMALL`).

## Giả thuyết

**H1 — Seed chung.** Gần cuối `generate.py`, code gốc gọi `set_seed(seed)` ngay
trước khi sinh *mỗi* lớp. Nhiễu ban đầu của ảnh thứ k vì thế giống hệt nhau ở mọi
lớp; chỉ prompt là khác. Nếu prompt yếu, ảnh gần như được quyết định bởi nhiễu.

**H2 — Prompt bị lấn át.** Guidance 2.0 vốn đã thấp, thêm LoRA kéo các lớp lại gần
nhau, nên tín hiệu "đây là lớp nào" không đủ mạnh để tách các lớp ra.

Hai giả thuyết không loại trừ nhau. Phép thử dưới đây tách được từng cái nhờ lưới
2 × 2: seed (chung / riêng) × guidance (2 / 5), cộng thêm SD gốc để đối chứng.

## Cách đo

Mỗi cấu hình sinh **20 ảnh mỗi lớp**, tức 200 ảnh, từ đúng LoRA đã train
(`shot16_seed0_tpl1/lr0.0001_epoch200/dataset-wise`). Không train lại gì.
Code sinh là `GenerateImage` của `generate.py`, chỉ đổi guidance và cách đặt seed.

Mốc so sánh là **bộ ảnh thật**: 20 ảnh mỗi lớp lấy ngẫu nhiên (seed 0) từ phần
`test` của file chia CoOp, không trùng với 16 ảnh few-shot.

| Chỉ số | Đo gì | Bình thường |
|---|---|---|
| `pix_same_k` | Tương quan pixel trung bình giữa ảnh thứ k của hai lớp khác nhau (ảnh xám 64×64) | ≈ 0 |
| `pix_gt08` | % cặp như trên có tương quan > 0,8, tức gần như bản sao | 0% |
| `acc_fewshot` | % ảnh mà CLIP xếp gần tâm của 16 ảnh thật đúng lớp nhất. Trả lời: *ảnh sinh có giống ảnh thật của đúng lớp đó không* | ảnh thật: 78,5% |
| `acc_zeroshot` | % ảnh CLIP zero-shot đoán đúng lớp, với prompt `"a centered satellite photo of …"` | ảnh thật: 54% |
| `inter_cos` | Cosine trung bình giữa tâm CLIP của hai lớp khác nhau trong cùng bộ; cao nghĩa là các lớp chụm lại | ảnh thật: 0,908 |

CLIP dùng bản **ViT-B/16**, cùng bản với bước classify. `acc_fewshot` là chỉ số
chính. `acc_zeroshot` chỉ để tham khảo, lý do ở phần kết quả.

## Kết quả

| Cấu hình | Seed | Guidance | `pix_same_k` | `pix_gt08` | `acc_fewshot` | `acc_zeroshot` | `inter_cos` |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Ảnh thật (mốc) | — | — | 0,004 | 0% | **78,5** | 54,0 | 0,908 |
| SD gốc | chung | 2 | 0,758 | 37% | 43,0 | 62,0 | 0,891 |
| SD gốc | riêng | 2 | 0,004 | 0% | 48,0 | 63,5 | 0,881 |
| DataDream (bộ đã sinh) | chung | 2 | **0,851** | **75%** | 43,5 | 36,5 | 0,934 |
| DataDream | riêng | 2 | 0,001 | 0% | 46,0 | 34,5 | 0,925 |
| DataDream | chung | 5 | 0,711 | 24% | **55,0** | 38,5 | 0,921 |
| DataDream | riêng | 5 | 0,002 | 0% | **53,0** | 41,5 | 0,922 |

Hàng "DataDream (bộ đã sinh)" sinh lại cho ra **đúng từng chỉ số** của 20 ảnh đầu
trong bộ 2.000 ảnh đã sinh, nên phép thử đo đúng thứ đã gặp lỗi.

Mỗi bộ chỉ có 200 ảnh, nên các tỉ lệ % sai số khoảng ±3,5 điểm (1 sai số chuẩn).
Chênh lệch dưới khoảng 7 điểm chưa đủ để kết luận.

### Đọc kết quả

**H1 đúng: seed chung gây trùng lặp, và chỉ gây trùng lặp.** Đổi sang seed riêng
thì `pix_same_k` về 0 ở mọi cấu hình, SD gốc cũng vậy. Nhưng `acc_fewshot` gần như
không đổi (43,5 → 46; 43 → 48, đều trong sai số). Seed chung làm các lớp *trông*
giống nhau; nó không phải lý do ảnh không giống lớp thật.

**H2 đúng: guidance là đòn bẩy cho nội dung.** Tăng guidance từ 2 lên 5 làm
`acc_fewshot` tăng khoảng 10 điểm (43,5 → 55; 46 → 53), khoảng 3 sai số chuẩn.
Cải thiện rõ nhất ở Industrial (11 → 20/20), Residential (14 → 20/20),
PermanentCrop (9 → 18/20) và AnnualCrop (7 → 11/20). Cái giá phải trả: guidance cao
thường làm ảnh trong cùng một lớp giống nhau hơn. Ở hình 3, HerbaceousVegetation và
SeaLake đã gần như một kiểu ảnh lặp lại. Phép đo này chỉ thưởng cho độ giống, không
phạt việc mất đa dạng, nên nó nghiêng về guidance cao.

**Theo chỉ số này, DataDream gần như không hơn SD gốc** (43,5 so với 43%, cả hai
cùng seed riêng thì 46 so với 48%). Đây chưa phải accuracy của classifier, xem
[So với paper](#so-với-paper).

**Không dùng `acc_zeroshot` để đánh giá DataDream.** Theo chỉ số này DataDream
*kém* SD gốc (35–41 so với 62–64%), nhưng chính ảnh thật cũng chỉ được 54%. Ảnh SD
gốc là "ảnh vệ tinh" theo nghĩa phổ thông, đúng kiểu mà prompt chữ của CLIP mô tả.
DataDream kéo ảnh về phong cách riêng của EuroSAT nên prompt chữ khó nhận ra hơn.
`acc_fewshot` so trực tiếp với ảnh thật nên đo đúng thứ cần đo hơn.

### Hình

Lưới 8 ảnh đầu mỗi lớp, mỗi lớp một hàng. Cột thứ k là ảnh thứ k của mọi lớp, nên
nếu các ô trong một cột giống nhau thì đó là dấu hiệu dùng chung seed.

| Hình | File |
|---|---|
| 1. DataDream, seed chung, guidance 2 (bộ đã sinh) | [diag_eurosat_dd_gs2_shared.jpg](../../results/figures/diag_eurosat_dd_gs2_shared.jpg) |
| 2. DataDream, seed riêng, guidance 2 | [diag_eurosat_dd_gs2_perclass.jpg](../../results/figures/diag_eurosat_dd_gs2_perclass.jpg) |
| 3. DataDream, seed riêng, guidance 5 | [diag_eurosat_dd_gs5_perclass.jpg](../../results/figures/diag_eurosat_dd_gs5_perclass.jpg) |
| 4. Ảnh thật (mốc) | [diag_eurosat_real.jpg](../../results/figures/diag_eurosat_real.jpg) |
| SD gốc, seed chung / riêng | [sd_gs2_shared](../../results/figures/diag_eurosat_sd_gs2_shared.jpg), [sd_gs2_perclass](../../results/figures/diag_eurosat_sd_gs2_perclass.jpg) |
| DataDream, seed chung, guidance 5 | [dd_gs5_shared](../../results/figures/diag_eurosat_dd_gs5_shared.jpg) |

Cặp hình 1 và 3 là cặp nên đưa vào báo cáo: ở hình 1 các cột trùng nhau, ở hình 3
mỗi hàng là một kiểu cảnh riêng.

## So với paper

Paper (mục thực nghiệm, Bảng 1) dùng guidance 2.0, 50 bước, **500 ảnh mỗi lớp**,
chạy 3 seed few-shot. Kết quả đo bằng **accuracy của CLIP ViT-B/16 sau khi train
LoRA** trên ảnh sinh, chấm trên tập test thật. EuroSAT, 16-shot:

| | Chỉ ảnh sinh | Thật + sinh |
|---|:-:|:-:|
| CLIP zero-shot | 38,1 | — |
| Chỉ ảnh thật (16-shot) | — | 93,5 ± 0,7 |
| IsSynth (SD không fine-tune) | 71,3 ± 2,9 | 93,9 ± 0,1 |
| DataDream cls | **85,4 ± 2,9** | 93,8 ± 0,3 |
| DataDream dset | 80,3 ± 4,1 | 93,4 ± 0,0 |

Từ bảng này rút ra được bốn điều:

- **Không so được 43–55% của ta với 80% của paper.** Paper đo accuracy sau khi
  train classifier. Ta mới đo ảnh sinh giống ảnh thật tới đâu, bằng CLIP chưa train.
  Một classifier train trên ảnh "hơi lệch" nhưng đa dạng vẫn có thể đạt accuracy cao.
- **Paper cũng chạy với lỗi seed chung**, vì đó là code gốc của chính họ. Con số
  80,3% của dset đã bao gồm ảnh hưởng của lỗi này.
- **Paper không thử guidance khác 2.0.** Họ chỉ nhắc, trong phần related work, rằng
  hạ guidance giúp ảnh đa dạng hơn (trích FakeIt). Tức là chọn 2.0 để ưu tiên độ đa
  dạng, không phải để ảnh giống nhất.
- **Biến thể dset là biến thể yếu hơn trên EuroSAT** (80,3 so với 85,4) và dao động
  nhiều nhất (±4,1). Bộ đã gặp lỗi chính là dset: một LoRA cho cả 10 lớp, nên dễ làm
  các lớp nhập vào nhau. Biến thể cls có một LoRA riêng cho mỗi lớp nên về cấu trúc
  không thể làm các lớp nhập vào nhau. Biến thể cls chưa được thử.

Kết luận: **giữ guidance 2.0**. Cách duy nhất để biết lỗi này có làm hỏng kết quả hay
không là train classifier ở bước 3, sau khi đã sửa known-issues #17. Bộ DataDream
EuroSAT dset (seed riêng, guidance 2) cho accuracy gần 80% thì lỗi đã hết; thấp hơn
hẳn thì khi đó mới thử guidance 5 như một mục ablation.

## Đã làm

- **Sửa `generate/generate.py`** (comment `NHOM SUA` ngay trước `generate_image.run`):
  `set_seed(seed)` đổi thành `set_seed(seed * 1000 + vị trí lớp trong SUBSET_NAMES)`.
  Lấy theo vị trí trong danh sách đầy đủ nên dù chia việc bằng `--n_set_split` thế
  nào, mỗi lớp vẫn nhận đúng một seed, và vẫn tái lập được. Nhân 1000 để
  `--seed=1` không trùng seed với `--seed=0`.
- **Đã kiểm bản sửa:** chạy `generate.py` đã sửa cho 10 ảnh SeaLake, kết quả giống
  từng pixel với ảnh "DataDream, seed riêng, guidance 2" của phép thử.
- **`scripts/check_env.py`** báo lỗi nếu ai đó lỡ đưa dòng `set_seed(seed)` cũ trở
  lại.
- **Guidance mặc định giữ nguyên 2.0**, theo paper. Chưa đổi.

## Chờ nhóm quyết

1. **Guidance cho các lần sinh chính thức.** Đề xuất: giữ 2.0 như paper (xem
   [So với paper](#so-với-paper)). Chỉ thử 5.0 như một mục ablation nếu bước 3 cho
   accuracy thấp hẳn so với paper. Nếu thử 5.0 thì SD gốc cũng phải sinh ở 5.0 để so
   công bằng. Guidance nằm trong tên thư mục ảnh sinh (`gs2.0_…`, `gs5.0_…`) nên hai
   bộ không ghi đè nhau.
2. **Sinh lại ba bộ đã sinh bằng seed chung.** Cả ba bộ vẫn nằm trên ổ, chưa bàn giao:

   | Bộ | Ảnh | Thời gian sinh lại |
   |---|:-:|:-:|
   | SD gốc Aircraft | 4.000 | ~1 giờ 45 |
   | SD gốc EuroSAT | 2.000 | ~55 phút |
   | DataDream EuroSAT | 2.000 | ~55 phút |

   Tên thư mục không chứa cách đặt seed, nên trước khi sinh lại phải đổi tên bộ cũ
   (ví dụ thêm `_sharedseed`). Nếu không, hàng đợi sẽ thấy đủ ảnh và bỏ qua.
3. **Fine-tune Aircraft** không phụ thuộc vào guidance, vì guidance chỉ dùng lúc
   sinh ảnh. Fine-tune đang chờ tập 16-shot Aircraft của bước 0, chưa có trên ổ.

## Còn mở

**Forest, SeaLake, River sai ở mọi cấu hình**, kể cả SD gốc:

- Forest bị xếp nhầm thành HerbaceousVegetation.
- SeaLake bị xếp nhầm thành Forest, hoặc River (với SD gốc).
- River bị xếp nhầm thành AnnualCrop.

Ở guidance 5, Pasture cũng tụt hẳn: chỉ còn 2–4/20 ảnh đúng, phần lớn bị xếp thành
PermanentCrop.

So hình 3 với hình 4 thì thấy lý do: ảnh thật EuroSAT có độ phân giải gốc 64×64,
tối và ít chi tiết. Rừng là một mảng xanh đậm, hồ là một mảng xanh đen. Ảnh sinh thì
sáng và nhiều chi tiết. Rừng thành cây thưa xen nhà, hồ thành mặt phẳng xám nhạt,
sông thành kênh giữa ruộng. LoRA 16-shot chưa kéo được ba lớp "đồng màu" này về
đúng phân phối.

Hướng có thể thử (chưa làm): guidance cao hơn nữa; prompt mô tả cụ thể hơn; train
LoRA ở 64×64 phóng to thay vì 512 để khớp độ nét; xem lại có cần bớt epoch không.
Phép đo `acc_fewshot` ở trên dùng được ngay để đánh giá từng hướng, khoảng 6 phút GPU
mỗi cấu hình.

## Hạn chế của phép đo

- Chỉ 20 ảnh mỗi lớp, sai số khoảng ±3,5 điểm cho mỗi tỉ lệ %.
- `acc_fewshot` là chỉ số thay thế, đo bằng CLIP chưa train. Nó **không phải** accuracy
  của classifier ở bước 3, chỉ cho biết ảnh sinh giống ảnh thật đúng lớp tới đâu.
- Chỉ một tập few-shot (`seed0`) và một LoRA.
- Mới thử trên EuroSAT. Aircraft chưa có LoRA. Riêng trùng lặp do seed chung thì đã
  thấy ở bộ SD gốc Aircraft: tương quan 0,79 giữa các lớp.

## Tái lập

Mỗi cấu hình mất khoảng 5–6 phút trên RTX 5070 Ti (200 ảnh, khoảng 1,6 giây/ảnh, kể
cả thời gian nạp model). Phần đo mất khoảng 1 phút. CLIP ViT-B/16 tải một lần
(~600 MB) vào `D:/models/clip-vit-base-patch16`.

```powershell
.venv\Scripts\activate
$L = "DataDream/outputs/eurosat/shot16_seed0_tpl1/lr0.0001_epoch200/dataset-wise"
$O = "synthetic/_diag/eurosat"

# sinh: 4 cấu hình DataDream + 2 cấu hình SD gốc (bỏ --lora)
python scripts\diag_generate.py $O/dd_gs2_shared   --lora $L --guidance 2 --seed_mode shared
python scripts\diag_generate.py $O/dd_gs2_perclass --lora $L --guidance 2 --seed_mode per_class
python scripts\diag_generate.py $O/dd_gs5_shared   --lora $L --guidance 5 --seed_mode shared
python scripts\diag_generate.py $O/dd_gs5_perclass --lora $L --guidance 5 --seed_mode per_class
python scripts\diag_generate.py $O/sd_gs2_shared   --guidance 2 --seed_mode shared
python scripts\diag_generate.py $O/sd_gs2_perclass --guidance 2 --seed_mode per_class

# do: in bang tren, ghi CSV va luoi anh
python scripts\diag_eval.py --csv results\diag_class_collapse.csv --sheet_dir results\figures `
    --sets dd_gs2_shared=$O/dd_gs2_shared dd_gs2_perclass=$O/dd_gs2_perclass `
           dd_gs5_shared=$O/dd_gs5_shared dd_gs5_perclass=$O/dd_gs5_perclass `
           sd_gs2_shared=$O/sd_gs2_shared sd_gs2_perclass=$O/sd_gs2_perclass
```

Tải CLIP (một lần):

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch16', allow_patterns=['*.json','*.txt','pytorch_model.bin'], local_dir='D:/models/clip-vit-base-patch16', local_dir_use_symlinks=False)"
```

Phải dùng `local_dir`. Nếu tải vào cache mặc định thì Windows báo `WinError 1314`
(không có quyền tạo symlink).
