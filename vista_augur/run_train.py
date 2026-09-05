"""第二工作点训练入口。"""

from __future__ import annotations

import argparse

from second_workpoint.config import load_config
from second_workpoint.training.trainer import build_trainer
from second_workpoint.utils.runtime import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="second workpoint torch-real training")
    parser.add_argument(
        "--config",
        type=str,
        default="vista_augur/configs/second_workpoint_deepcorr300_torch_real_smoke.json",
        help="训练配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.random_seed)

    trainer = build_trainer(config)
    if config.is_training:
        summary = trainer.train()
        print(f"[done] training finished, summary saved under {summary['setting_name']}")
    else:
        summary = trainer.evaluate()
        print(f"[done] eval summary: {summary}")


if __name__ == "__main__":
    main()