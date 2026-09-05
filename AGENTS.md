# AGENTS.md

## Scope

This file applies to the repository root and all subdirectories.

## Project Goal

This repository contains the second work point of the Augur traffic-correlation-defense project. The objective is to generate physically deployable future traffic perturbations from historical Tor traffic while reducing the effectiveness of frozen correlation-attack models.

Preserve continuity with the first work point: the research contribution should come from the generator representation and reprogramming mechanism, not from silently changing datasets, target models, or evaluation rules.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| `vista_augur/second_workpoint/` | Trainable model, data adapters, loss, and trainer |
| `vista_augur/configs/` | Reproducible experiment configurations |
| `vista_augur/docs/` | Technical designs, runbooks, and experiment records |
| `datasets/` | Local DeepCorr and DeepCoFFEA datasets; ignored by Git |
| `target_model/` | Frozen target-model code and local weights |
| `base_models/` | Local frozen LLM backbones; ignored by Git |
| `vista_augur/outputs/` | Checkpoints and logs; ignored by Git |

Run commands from the repository root. The Python import root is `vista_augur/`.

## Research Constraints

- Reuse the datasets under `datasets/` unless the user explicitly requests a new dataset.
- Keep comparisons with the first work point fair: use the same data split, target model, window settings, perturbation constraints, and evaluation protocol whenever possible.
- The output must remain a future traffic perturbation, not a traffic classifier or anomaly score.
- Time and size perturbations must respect non-negative, additive, and deployable physical constraints.
- Distinguish semantic roles precisely: time-series and visual features require mapping into the LLM-compatible space; prompt text is already in language space and acts as a semantic anchor.
- Do not claim semantic alignment unless the implemented mechanism and experiment directly support that claim.

## Implementation Rules

- Prefer the smallest change that cleanly satisfies the task.
- Reuse existing modules and configuration fields before adding abstractions.
- Do not add dependencies unless explicitly requested or technically unavoidable.
- Keep the frozen Qwen backbone and frozen target model differentiable with respect to generator inputs.
- Keep machine-specific paths, credentials, datasets, model weights, checkpoints, logs, caches, and PDFs out of Git.
- Preserve unrelated user changes in a dirty worktree.

## Documentation Rules

- Write new project documents under `vista_augur/docs/`.
- Write repository files and project documentation in English. User-facing explanations may be in Chinese.
- For translation-only tasks, use the lightest available model that can preserve technical meaning and formatting.
- Explain technical decisions through mechanism, necessity, and experimental effect rather than feature lists.
- Use tables, Mermaid diagrams, formulas, or pseudocode when they materially improve clarity.
- Keep comments, docstrings, parameter descriptions, and error messages in document code examples in Chinese.
- Maintain a precise, restrained tone and clearly separate verified results from hypotheses.

## Verification

Before claiming completion, run checks proportional to the change:

1. Confirm configuration and referenced resource paths.
2. Run Python import and `compileall` checks.
3. Run targeted tests or smoke checks when the required local models and hardware are available.
4. Search for stale paths after moving or renaming files.
5. Before committing, inspect staged files and confirm that no large binaries or credentials are included.

If end-to-end training cannot be run locally, state the missing environment or model dependency explicitly instead of claiming full runtime verification.

## Remote Experiments

Local code may be synchronized to the experiment server and executed there. Keep local and remote directory layouts consistent, synchronize only intended files, avoid overwriting unrelated remote work, and bring back logs or summaries needed for analysis. Never store SSH passwords or other credentials in repository files.

Use the following connection profile for the current experiment server:

| Field | Value |
| --- | --- |
| Protocol | SSH |
| Host | `100.79.197.115` |
| Port | `22` |
| User | `xilin` |
| Remote workspace | `/home/xilin/work2` |
| Authentication | Supply the password interactively or through an approved credential manager; never place it in commands, scripts, logs, or tracked files |

The remote workspace exists and contains the synchronized project resources. Recheck its contents before every synchronization because remote state may have changed.
