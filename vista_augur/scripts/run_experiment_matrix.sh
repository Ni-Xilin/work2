#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

group="${1:-smoke}"
python_bin="${VISTA_PYTHON:-/home/xilin/anaconda3/envs/deepcorr/bin/python}"
visible_devices="${VISTA_CUDA_VISIBLE_DEVICES:-0,1}"
log_dir="vista_augur/outputs/logs"
mkdir -p "$log_dir"

case "$group" in
    smoke)
        configs=(
            second_workpoint_deepcorr300_torch_real_smoke.json
            second_workpoint_mdeepcorr_smoke.json
            second_workpoint_deepcoffea_real_smoke.json
        )
        ;;
    primary)
        configs=(
            second_workpoint_deepcorr300_torch_real_probe_text_prototypes.json
            second_workpoint_deepcorr300_torch_real_tune_text_prototypes.json
            second_workpoint_deepcorr300_work1_aligned.jsonc
        )
        ;;
    ablation)
        configs=(
            second_workpoint_deepcorr300_ablation_time_only.json
            second_workpoint_deepcorr300_ablation_size_only.json
            second_workpoint_deepcorr300_ablation_no_visual.json
            second_workpoint_deepcorr300_ablation_no_prompt.json
            second_workpoint_deepcorr300_torch_real_smoke_learned.json
        )
        ;;
    cross-target)
        configs=(
            second_workpoint_mdeepcorr_full.json
            second_workpoint_deepcoffea_full.json
        )
        ;;
    *)
        echo "Usage: $0 {smoke|primary|ablation|cross-target}" >&2
        exit 2
        ;;
esac

for config_name in "${configs[@]}"; do
    timestamp="$(date +%Y%m%d-%H%M%S)"
    log_path="$log_dir/${config_name%.*}-$timestamp.log"
    echo "Running $config_name; log=$log_path"
    CUDA_VISIBLE_DEVICES="$visible_devices" PYTHONPATH=vista_augur \
        "$python_bin" vista_augur/run_train.py \
        --config "vista_augur/configs/$config_name" 2>&1 | tee "$log_path"
done
