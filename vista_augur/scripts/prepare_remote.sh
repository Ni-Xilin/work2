#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${VISTA_PYTHON:-/home/xilin/anaconda3/envs/deepcorr/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    echo "Set VISTA_PYTHON to an existing Python executable." >&2
    exit 1
fi

PYTHONPATH=vista_augur "$python_bin" -m unittest discover -s vista_augur/tests -v

if [[ ! -f base_models/Qwen2.5-1.5B-Instruct/config.json ]]; then
    echo "Missing base_models/Qwen2.5-1.5B-Instruct." >&2
    echo "Download it on an Internet-connected machine and synchronize the directory." >&2
    exit 1
fi

echo "Environment ready: $python_bin"
echo "Backbone snapshot found: base_models/Qwen2.5-1.5B-Instruct"
