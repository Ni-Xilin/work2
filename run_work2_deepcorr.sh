#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${project_root}/vista_augur/configs/second_workpoint_deepcorr300_work1_aligned.jsonc"

cd "${project_root}"
exec python3 vista_augur/run_work2.py "${config_path}"
