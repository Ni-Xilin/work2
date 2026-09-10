"""Export complete Work2 mDeepCorr cascade evaluation data."""

from pathlib import Path
import sys


EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common import EvaluationProfile, run


PROFILE = EvaluationProfile(
    name="mdeepcorr",
    config="vista_augur/configs/mdeepcorr_config.jsonc",
    checkpoint="auto",
    model_dir="eval/mdeepcorr",
    target_names=("mdeepcorr", "mdeepcorrtorch"),
    protocol="mdeepcorr_dc100_gt_0.01_then_dc700",
    batch_size=16,
    num_workers=32,
    shuffle=False,
    drop_last=True,
    scoring_batch_size=64,
)


if __name__ == "__main__":
    run(PROFILE)
