"""Export complete Work2 DeepCoFFEA correlation-matrix evaluation data."""

from pathlib import Path
import sys


EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common import EvaluationProfile, run


PROFILE = EvaluationProfile(
    name="deepcoffea",
    config="vista_augur/configs/deepcoffea_config.jsonc",
    checkpoint="auto",
    model_dir="eval/deepcoffea",
    target_names=("deepcoffea", "deepcoffeatorch"),
    protocol="deepcoffea_matrix",
    batch_size=25,
    num_workers=0,
    shuffle=True,
    drop_last=True,
    max_steps=20,
    scoring_batch_size=275,
)


if __name__ == "__main__":
    run(PROFILE)
