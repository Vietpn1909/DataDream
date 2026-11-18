# Hang doi GPU chay qua dem - TU TIEP TUC: lo tat may thi chay lai DUNG lenh nay.
# Moi dem: sua danh sach $jobs ben duoi. Log va thoi gian: outputs_logs/.
#
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_queue.ps1
#   them -DryRun de chi in ra se lam gi, khong chay
#
# - Bo anh nao du roi thi bo qua. Bo anh do dang thi chi sinh cac lop con thieu;
#   lop dang do duoc sinh LAI TU ANH 0 (mat toi da ~6 phut), xem ghi chu o vong lap.
# - Fine-tune da xong (co pytorch_lora_weights.safetensors) thi bo qua; dang do
#   va con checkpoint thi tu them --resume_from_checkpoint=latest. Thieu co nay
#   la mat that: no train lai tu dau va xoa checkpoint cu (limit = 1).
# - Giu may khong ngu trong luc chay (het hieu luc khi script ket thuc).
# - Tu choi chay neu dang co generate.py / datadream.py khac chay.
#   them -Only <tien to ten viec>[,...] de chi chay mot phan, vd -Only aircraft20_issynth,eurosat_issynth
param([switch]$DryRun, [string[]]$Only)

$repo = Split-Path -Parent $PSScriptRoot
$py = "$repo\.venv\Scripts\python.exe"
$acc = "$repo\.venv\Scripts\accelerate.exe"
$status = Join-Path $PSScriptRoot "queue_status.py"
$logDir = "$repo\outputs_logs"
$summary = "$logDir\queue_summary.tsv"
# Phai trung voi save_dir trong generate/local.yaml - hang doi dem anh o day de
# biet bo nao con thieu.
$synth = "$repo\synthetic"
New-Item -ItemType Directory -Force $logDir | Out-Null
$commit = (git -C $repo rev-parse --short HEAD).Trim()

$busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "generate\.py|datadream\.py|issynth\.py" }
if ($busy) {
    "DANG CO TIEN TRINH KHAC CHAY - khong chay chong len:"
    $busy | ForEach-Object { "  PID $($_.ProcessId): $($_.CommandLine)" }
    if (-not $DryRun) { exit 1 }
}

$gen = @("generate.py", "--sd_version=sd2.1", "--n_template=1", "--n_img_per_class=200",
         "--guidance_scale=2.0", "--bs=10", "--n_set_split=1", "--split_idx=0")
$common = @("--fewshot_seed=seed0", "--n_shot=16", "--n_template=1",
            "--train_text_encoder=True", "--train_batch_size=8",
            "--gradient_accumulation_steps=1", "--learning_rate=1e-4",
            "--lr_scheduler=cosine", "--lr_warmup_steps=100", "--num_train_epochs=200",
            "--mixed_precision=bf16", "--checkpointing_steps=1000",
            "--checkpoints_total_limit=1", "--report_to=tensorboard", "--output_dir=outputs")
$dd = @("--mode=datadream", "--n_shot=16", "--fewshot_seed=seed0",
        "--datadream_lr=1e-4", "--datadream_epoch=200")

$aircraftOut = "$repo\DataDream\outputs\fgvc_aircraft"
$aircraftSyn = "$synth\fgvc_aircraft\sd2.1\gs2.0_nis50"
$ablN = @("--n_img_per_class=100")  # bo ablation: 100 anh/lop (docs/run.md); co sau thang co truoc

# Da chot (18/11/2026): IsSynth strength 0.15; ablation K = 1, 4 va chi U-Net, deu la dset.
$jobs = @(
    @{ name = "aircraft20_issynth"; type = "iss"; dataset = "fgvc_aircraft"; nclass = 20; n = 200
       out = "$aircraftSyn\shot16_seed0_template1_issynth_str0.15"
       args = @("issynth.py", "--dataset=fgvc_aircraft", "--strength=0.15", "--n_img_per_class=200",
                "--fewshot_dir=$repo\data\fgvc_aircraft\real_train_fewshot") },
    @{ name = "eurosat_issynth"; type = "iss"; dataset = "eurosat"; nclass = 10; n = 200
       out = "$synth\eurosat\sd2.1\gs2.0_nis50\shot16_seed0_template1_issynth_str0.15"
       args = @("issynth.py", "--dataset=eurosat", "--strength=0.15", "--n_img_per_class=200",
                "--fewshot_dir=$repo\data\eurosat\real_train_fewshot") },
    @{ name = "aircraft20_dset_K4_finetune"; type = "ft"
       ckdir = "$aircraftOut\shot4_seed0_tpl1\lr0.0001_epoch200\dataset-wise"
       args = @("launch", "datadream.py", "--dataset=fgvc_aircraft") + $common + @("--n_shot=4") },
    @{ name = "aircraft20_dset_K4_generate"; type = "gen"; dataset = "fgvc_aircraft"; nclass = 20; n = 100
       out = "$aircraftSyn\shot4_seed0_template1_lr0.0001_ep200_dswise"
       args = $gen + $dd + @("--is_dataset_wise_model=True", "--dataset=fgvc_aircraft", "--n_shot=4") + $ablN },
    @{ name = "aircraft20_dset_K1_finetune"; type = "ft"
       ckdir = "$aircraftOut\shot1_seed0_tpl1\lr0.0001_epoch200\dataset-wise"
       args = @("launch", "datadream.py", "--dataset=fgvc_aircraft") + $common + @("--n_shot=1") },
    @{ name = "aircraft20_dset_K1_generate"; type = "gen"; dataset = "fgvc_aircraft"; nclass = 20; n = 100
       out = "$aircraftSyn\shot1_seed0_template1_lr0.0001_ep200_dswise"
       args = $gen + $dd + @("--is_dataset_wise_model=True", "--dataset=fgvc_aircraft", "--n_shot=1") + $ablN },
    @{ name = "aircraft20_dset_unetonly_finetune"; type = "ft"
       ckdir = "$aircraftOut\shot16_seed0_tpl1_notextlora\lr0.0001_epoch200\dataset-wise"
       args = @("launch", "datadream.py", "--dataset=fgvc_aircraft") + $common + @("--train_text_encoder=False") },
    @{ name = "aircraft20_dset_unetonly_generate"; type = "gen"; dataset = "fgvc_aircraft"; nclass = 20; n = 100
       out = "$aircraftSyn\shot16_seed0_template1_lr0.0001_ep200_notextlora_dswise"
       args = $gen + $dd + @("--is_dataset_wise_model=True", "--dataset=fgvc_aircraft",
                             "--datadream_train_text_encoder=False") + $ablN }
)

if ($Only) {
    # powershell -File truyen "a,b" thanh MOT chuoi, nen tach lai theo dau phay
    $Only = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_.Trim() })
    $jobs =@($jobs | Where-Object { $n = $_.name; @($Only | Where-Object { $n -like "$_*" }).Count -gt 0 })
    "CHI CHAY: " + (($jobs | ForEach-Object { $_.name }) -join ", ")
}

# Thong bao trong ham dung Write-Host: moi gia tri khac in ra deu bi gop vao gia
# tri tra ve, lam $rc thanh mang thay vi ma thoat.
function Run-Step($name, $exe, $argv, $cwd, $note) {
    $stamp = (Get-Date).ToString("MMdd-HHmmss")
    $start = Get-Date
    Write-Host ("  [{0}] chay: {1}" -f $start.ToString("HH:mm:ss"), $note)
    if ($DryRun) { return 0 }
    $p = Start-Process $exe -ArgumentList $argv -WorkingDirectory $cwd -PassThru -NoNewWindow -Wait `
        -RedirectStandardOutput "$logDir\$name.$stamp.out.log" -RedirectStandardError "$logDir\$name.$stamp.err.log"
    $end = Get-Date
    if (-not (Test-Path $summary)) { "job`tstart`tend`tminutes`texit`tcommit`tnote`tcommand" | Out-File -Encoding utf8 $summary }
    "{0}`t{1}`t{2}`t{3:N1}`t{4}`t{5}`t{6}`t{7}" -f $name, $start.ToString("s"), $end.ToString("s"),
        ($end - $start).TotalMinutes, $p.ExitCode, $commit, $note, ((Split-Path $exe -Leaf) + " " + ($argv -join " ")) |
        Out-File -Encoding utf8 -Append $summary
    Write-Host ("  [{0}] xong: exit {1}, {2:N1} phut" -f $end.ToString("HH:mm:ss"), $p.ExitCode, ($end - $start).TotalMinutes)
    return $p.ExitCode
}

Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
if (-not $DryRun) { [Win.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null }

$failed = $false
foreach ($j in $jobs) {
    "=== $($j.name) ==="
    if ($j.type -eq "ft") {
        if (Test-Path "$($j.ckdir)\pytorch_lora_weights.safetensors") { "  da xong - bo qua"; continue }
        $argv = $j.args
        $note = "tu dau"
        $ck = Get-ChildItem $j.ckdir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue
        if ($ck) { $argv = $argv + @("--resume_from_checkpoint=latest"); $note = "tiep tuc tu " + (($ck | ForEach-Object Name) -join ",") }
        $rc = Run-Step $j.name $acc $argv "$repo\DataDream" $note
        if ($rc -ne 0 -or (-not $DryRun -and -not (Test-Path "$($j.ckdir)\pytorch_lora_weights.safetensors"))) { $failed = $true; break }
    }
    elseif ($j.type -eq "iss") {
        # issynth.py tu bo qua lop da du, tu sinh lai lop dang do tu anh 0, va dung neu thu muc
        # da sinh voi tham so khac (issynth_meta.json) -> chi can truyen cac lop con thieu.
        $extra = @(); if ($DryRun) { $extra = @("--no-delete") }
        $todo = @(& $py $status $j.out $j.dataset $j.n @extra)
        if ($todo.Count -eq 0) { "  du $($j.n) anh x $($j.nclass) lop - bo qua"; continue }
        $idxs = @($todo | ForEach-Object { ($_ -split "`t")[0] })
        "  con thieu $($todo.Count)/$($j.nclass) lop"
        $rc = Run-Step $j.name $py ($j.args + @("--classes") + $idxs) "$repo\generate" ("lop " + ($idxs -join ","))
        if ($rc -ne 0) { $failed = $true; break }
        if (-not $DryRun) {
            $left = @(& $py $status $j.out $j.dataset $j.n)
            if ($left.Count -gt 0) { "  VAN CON THIEU sau khi chay: $left"; $failed = $true; break }
        }
    }
    else {
        $extra = @(); if ($DryRun) { $extra = @("--no-delete") }
        $todo = @(& $py $status $j.out $j.dataset $j.n @extra)
        if ($todo.Count -eq 0) { "  du $($j.n) anh x $($j.nclass) lop - bo qua"; continue }
        "  con thieu $($todo.Count)/$($j.nclass) lop"
        foreach ($line in $todo) {
            $idx, $cls, $cs = $line -split "`t"
            # Khong sinh tiep tu anh $cs: generate.py dat lai seed o dau moi lop nen
            # anh $cs se nhan dung nhieu cua anh 0 -> ban sao y het trong cung lop.
            # Sinh lai ca lop tu 0 thi ket qua giong het lan chay khong bi ngat.
            $argv = $j.args + @("--n_set_split=$($j.nclass)", "--split_idx=$idx", "--count_start=0")
            $note = "lop $idx ($cls) tu anh 0"
            if ([int]$cs -gt 0) { $note += " (da co $cs anh, sinh lai ca lop)" }
            $rc = Run-Step $j.name $py $argv "$repo\generate" $note
            if ($rc -ne 0) { $failed = $true; break }
        }
        if ($failed) { break }
        if (-not $DryRun) {
            $left = @(& $py $status $j.out $j.dataset $j.n)
            if ($left.Count -gt 0) { "  VAN CON THIEU sau khi chay: $left"; $failed = $true; break }
        }
    }
}
if (-not $DryRun) { [Win.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null }
if ($failed) { "DUNG LAI vi loi - xem log trong $logDir"; exit 1 }
"HET HANG DOI"
