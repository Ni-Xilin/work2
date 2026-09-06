"""Command-line entry point for second-work-point training and evaluation."""

from __future__ import annotations

import argparse

from second_workpoint.config import load_config
from second_workpoint.training.trainer import build_trainer
from second_workpoint.utils.runtime import save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="second workpoint torch-real training")
    parser.add_argument(
        "--config",
        type=str,
        default="vista_augur/configs/deepcorr_config.jsonc",
        help="Path to the experiment JSON configuration.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint, overriding resume_from_checkpoint.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation only, overriding is_training in the configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default=None,
        help="Override the evaluation split. --evaluate defaults to final_eval_split.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume is not None:
        config.resume_from_checkpoint = args.resume
    if args.evaluate:
        config.is_training = 0
        config.eval_split = args.eval_split or config.final_eval_split
    elif args.eval_split is not None:
        config.eval_split = args.eval_split
    if args.seed is not None:
        config.random_seed = args.seed
    config.validate()
    seed_everything(config.random_seed)

    trainer = build_trainer(config)
    if config.is_training:
        summary = trainer.train()
        print(f"[done] training finished, summary saved under {summary['setting_name']}")
    else:
        summary = trainer.evaluate()
        save_json(trainer.run_dir / "evaluation_summary.json", summary)
        print(f"[done] eval summary: {summary}")


if __name__ == "__main__":
    main()
