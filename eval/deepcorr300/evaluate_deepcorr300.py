"""Export complete Work2 DeepCorr300 evaluation data."""

from pathlib import Path
import sys


EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common import EvaluationProfile, run


PROFILE = EvaluationProfile(
    name="deepcorr300",
    config="vista_augur/configs/deepcorr_config.jsonc",
    checkpoint="auto",
    model_dir="eval/deepcorr300",
    target_names=("deepcorr300", "deepcorr"),
    protocol="deepcorr_1_to_199",
    batch_size=16,
    num_workers=0,
    shuffle=False,
    drop_last=True,
    scoring_batch_size=256,
)


if __name__ == "__main__":
    run(PROFILE)
