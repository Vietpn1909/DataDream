# Hang doi GPU chay qua dem - TU TIEP TUC: lo tat may thi chay lai DUNG lenh nay.
# Moi dem: sua danh sach $jobs ben duoi. Log va thoi gian: outputs_logs/.
#
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_queue.ps1
#   them -DryRun de chi in ra se lam gi, khong chay
#
# - Bo anh nao du roi thi bo qua. Bo anh do dang thi chi sinh cac lop con thieu,
#   tu dung so anh con thieu (anh PNG ghi do khi mat dien bi phat hien va xoa).
# - Fine-tune da xong (co pytorch_lora_weights.safetensors) thi bo qua; dang do
#   va con checkpoint thi tu them --resume_from_checkpoint=latest. Thieu co nay
#   la mat that: no train lai tu dau va xoa checkpoint cu (limit = 1).
# - Giu may khong ngu trong luc chay (het hieu luc khi script ket thuc).
# - Tu choi chay neu dang co generate.py / datadream.py khac chay.
param([switch]$DryRun)

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
    Where-Object { $_.CommandLine -match "generate\.py|datadream\.py" }
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

$eurosatCls = @("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake")
$ckRoot = "$repo\DataDream\outputs\eurosat\shot16_seed0_tpl1\lr0.0001_epoch200"

$jobs = @(
    @{ name = "eurosat_sdbase"; type = "gen"; dataset = "eurosat"; nclass = 10; n = 200
       out = "$synth\eurosat\sd2.1\gs2.0_nis50\shot0_template1"
       args = $gen + @("--mode=zeroshot", "--dataset=eurosat") },
    @{ name = "eurosat_dset_finetune"; type = "ft"
       ckdir = "$ckRoot\dataset-wise"
       args = @("launch", "datadream.py", "--dataset=eurosat") + $common },
    @{ name = "eurosat_dset_generate"; type = "gen"; dataset = "eurosat"; nclass = 10; n = 200
       out = "$synth\eurosat\sd2.1\gs2.0_nis50\shot16_seed0_template1_lr0.0001_ep200_dswise"
       args = $gen + $dd + @("--is_dataset_wise_model=True", "--dataset=eurosat") }
)
# cls: mot LoRA moi lop, 400 buoc - qua it de ghi checkpoint, tat may giua chung
# thi lop do train lai tu dau (~5 phut), cac lop da xong duoc bo qua
for ($i = 0; $i -lt $eurosatCls.Count; $i++) {
    $jobs += @{ name = "eurosat_cls_finetune_$i"; type = "ft"; ckdir = "$ckRoot\$($eurosatCls[$i])"
                args = @("launch", "datadream.py", "--dataset=eurosat", "--target_class_idx=$i") + $common }
}
$jobs += @(
    @{ name = "eurosat_cls_generate"; type = "gen"; dataset = "eurosat"; nclass = 10; n = 200
       out = "$synth\eurosat\sd2.1\gs2.0_nis50\shot16_seed0_template1_lr0.0001_ep200"
       args = $gen + $dd + @("--is_dataset_wise_model=False", "--dataset=eurosat") },
    @{ name = "aircraft20_sdbase"; type = "gen"; dataset = "fgvc_aircraft"; nclass = 20; n = 200
       out = "$synth\fgvc_aircraft\sd2.1\gs2.0_nis50\shot0_template1"
       args = $gen + @("--mode=zeroshot", "--dataset=fgvc_aircraft") }
)

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
    else {
        $extra = @(); if ($DryRun) { $extra = @("--no-delete") }
        $todo = @(& $py $status $j.out $j.dataset $j.n @extra)
        if ($todo.Count -eq 0) { "  du $($j.n) anh x $($j.nclass) lop - bo qua"; continue }
        "  con thieu $($todo.Count)/$($j.nclass) lop"
        foreach ($line in $todo) {
            $idx, $cls, $cs = $line -split "`t"
            $argv = $j.args + @("--n_set_split=$($j.nclass)", "--split_idx=$idx", "--count_start=$cs")
            $rc = Run-Step $j.name $py $argv "$repo\generate" ("lop $idx ($cls) tu anh $cs")
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
