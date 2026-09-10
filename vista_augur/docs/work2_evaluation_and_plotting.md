# Work2 Evaluation and Plotting Guide

## 1. Purpose and evidence base

This directory implements a single reproducible evaluation path for the second work point. Its scope is derived from:

- `Time-LLM-ICLR-2024.pdf`: frozen-LLM reprogramming, training dynamics, component ablation, and efficiency analysis;
- `TDSC-2026-03-0884-Main Document.pdf` and its appendices: low-FPR attack evaluation, ROC curves, correlation-score matrices, traffic overhead, runtime efficiency, component ablation, adaptive attacks, and transferability;
- `work1_reference/Generator_Trainer/`: the original DeepCorr, m-DeepCorr, and DeepCoFFEA protocols;
- `N:/postgraduate/小论文相关/codetest-main/edit_code1/`: the revised low-FPR, adaptive-attack, RegulaTor, overhead, and consistency-check scripts.

The implementation preserves Work1's dataset, split, target model, negative-pair construction, and low-FPR threshold rule. It does not treat forecasting MSE/MAE from Time-LLM as defense metrics: Work2 still outputs a deployable future perturbation, so attack degradation and physical cost are the primary outcomes.

## 2. Evaluation contract

`python vista_augur/run_train.py --evaluate ...` now writes `evaluation_scores.npz` beside `evaluation_summary.json`. The NPZ contains:

```text
clean_positive_scores
clean_negative_scores
adv_positive_scores
adv_negative_scores
```

The offline evaluator also accepts legacy Work1 pickle files containing `(all_outputs, all_labels)`, dictionaries containing `all_outputs` and `all_labels`, adaptive-attack dictionaries whose values use those two keys, and DeepCoFFEA correlation matrices.

Every experiment is described in a JSON manifest. Copy `data/manifest.example.json`, replace the placeholder run path, and add one record per target, baseline, ablation, adaptive attack, or transfer experiment. The evaluator creates one `*.summary.json`, one `*.curves.npz`, and a shared `index.json`:

```powershell
python -m data.evaluate --manifest data/my_experiments.json --output-dir data/results
python -m data.validate_results data/results/*.summary.json
```

For a legacy DeepCoFFEA matrix, use:

```json
{
  "name": "work2_deepcoffea",
  "target": "DeepCoFFEA",
  "method": "Work2",
  "artifact": "path/to/corrmatrix.npz",
  "format": "correlation_matrix",
  "matrix_key": "corr_matrix",
  "n_windows": 11,
  "vote_threshold": 9,
  "fpr_targets": [0.001, 0.0001]
}
```

The DeepCoFFEA adapter converts voting to a scalar session score exactly: accepting at least `k` windows at threshold `eta` is equivalent to thresholding the `k`-th largest window score. This avoids the slow threshold loop in the legacy scripts while preserving its decision rule.

## 3. Metrics and threshold semantics

For positive scores \(S^+\), negative scores \(S^-\), and threshold \(\tau\):

\[
\mathrm{FPR}=\frac{FP}{FP+TN},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},\quad
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
F_1=\frac{2PR}{P+R}.
\]

To match Work1, scores are sorted in descending order and the first negative rank reaching the requested FPR is selected. At least \(1/\mathrm{FPR}\) negatives are required to resolve that operating point. A report with fewer samples is marked `resolvable: false`; `validate_results.py` fails by default so an unsupported \(10^{-4}\) result cannot silently enter a paper.

Each defended condition reports two interpretations:

- `operating_points`: recalibrate the threshold on that condition, matching the Work1 paper tables and ROC protocol;
- `fixed_clean_threshold`: reuse the clean-data threshold, exposing deployment-time threshold drift.

ROC AUC, average precision, confusion counts, score-distribution summaries, actual achieved FPR, threshold, and sample counts are retained for auditability. Optional stratified bootstrap confidence intervals are enabled with `"bootstrap": 1000` in a manifest record.

## 4. Physical perturbation evaluation

Add an `overhead_artifact` NPZ to an experiment record when packet-level arrays are available. It must contain `original` and `adversarial` arrays shaped `(..., channels, length)`; an optional `valid_mask` excludes padding. Specify `time_channels`, `size_channels`, and unit conversions in the manifest.

The evaluator reports:

- relative time and size \(L_2\) ratios, preserving Table II comparability;
- modified-value fraction, mean/median/P90/P95/P99/max positive delay and padding;
- total added delay and size, relative \(L_1\) traffic overhead, and the corresponding Work1 Table IV quantities;
- negative time/size deltas, direction flips, and non-finite values as physical-constraint violations.

Signed traffic channels are compared by magnitude, so a negative packet-size direction marker is not mistaken for a negative physical size. A valid Work2 perturbation must add magnitude without changing direction.

Efficiency measurements are intentionally separated from traffic delay. `data.benchmark` measures any zero-argument callable, or a factory that returns one, after configurable warm-up iterations. It records wall-clock latency, throughput, process CPU utilization, and peak resident memory without adding a profiler dependency:

```powershell
python -m data.benchmark --factory my_benchmark:build_generator_call --factory-kwargs '{"config":"vista_augur/configs/deepcorr_config.jsonc"}' --warmup 10 --repetitions 100 --items-per-call 16 --output data/measurements/deepcorr.npz
```

The factory must perform setup once and return the zero-argument inference closure to measure. This prevents model loading from contaminating per-call inference latency. Put any additional repeated measurements, such as one-time `training_seconds`, in the same NPZ and run:

```powershell
python -m data.evaluate_efficiency --input data/measurements/deepcorr.npz --output data/results/deepcorr_efficiency.json
```

This separation follows the asynchronous deployment design in Work1: model inference latency is a compute metric, whereas generated packet delay is a traffic-overhead metric.

## 5. Figure inventory

All plotting scripts use a non-interactive backend and save publication-ready PDF/SVG/PNG files.

关于 `plot/` 下每个文件、输入格式、前置文件生成命令和当前限制的中文说明，见 [`plot_file_reference_zh.md`](plot_file_reference_zh.md)。

| Paper purpose | Script | Input |
| --- | --- | --- |
| Main effectiveness ROC panels | `plot.plot_roc` | summary JSON + curve NPZ |
| PRE/REC/F1 at low FPR | `plot.plot_operating_points` | summary JSON |
| Correlation suppression | `plot.plot_score_matrix` | NPZ square matrices |
| Relative and applied overhead | `plot.plot_overhead` | summaries with overhead |
| Training convergence and perturbation strategy | `plot.plot_training` | `train_summary.json` |
| CPU/runtime/training efficiency | `plot.plot_efficiency` | efficiency JSON |
| Time-only, size-only, branch/prompt/reprogramming ablations | `plot.plot_ablation` | ablation summaries |
| Black-box transfer and adaptive attacker | `plot.plot_transferability` | transfer/adaptive summaries |

Examples:

```powershell
python -m plot.plot_roc data/results/*.summary.json --output plot/results/main_roc.pdf
python -m plot.plot_operating_points data/results/*.summary.json --fpr fpr_1e-04 --output plot/results/f1_1e-4.pdf
python -m plot.plot_training vista_augur/outputs/checkpoints/*/train_summary.json --output plot/results/training.pdf
python -m plot.plot_score_matrix --input data/matrices/deepcorr.npz --keys clean_matrix adv_matrix --labels Clean Work2 --output plot/results/deepcorr_matrix.pdf
```

For a reproducible full figure build, copy `plot/figures.example.json`, list every required figure, then run:

```powershell
python -m plot.plot_all --manifest plot/my_figures.json
```

## 6. Required experiment matrix

The following runs are needed before claiming a complete paper evaluation:

1. Main white-box evaluation for DeepCorr-300, m-DeepCorr 100-to-700, and DeepCoFFEA on the unchanged Work1 test splits.
2. No-perturbation and relevant Work1 baselines under the same negative-pair and FPR protocol.
3. Time-only, size-only, and time-plus-size perturbation runs.
4. Work2 component ablations: temporal branch, visual branch, prompt text, and learned-vs-text prototypes. A disabled component must be represented by a separately trained checkpoint; post-hoc masking is not equivalent.
5. Adaptive attacks trained on Work2-perturbed traffic and evaluated on both perturbed and clean traffic.
6. Strict black-box transfer in which the generator checkpoint is trained on one surrogate and evaluated against an unseen target without further fitting.
7. Packet-level overhead, repeated inference timing, peak RAM/CPU, and one-time training duration.
8. At least three random seeds for reported Work2 comparisons when compute permits; aggregate confidence intervals rather than presenting one favorable run.

## 7. Interpretation rules

- Lower defended TPR/F1/AUC means stronger defense; higher clean attack performance confirms the frozen attacker and data protocol are functioning.
- Do not compare low-FPR values unless `resolvable` is true and the negative sample counts match the intended protocol.
- Do not call a result semantic alignment unless the text-prototype or prompt mechanism is directly ablated against a valid alternative.
- Do not claim deployability from relative \(L_2\) alone. Physical-constraint checks and applied delay/size overhead must also pass.
- Do not mix independently recalibrated and fixed-clean-threshold results in one table without explaining that they answer different questions.
- Generated figures are outputs, not source data. Preserve summary JSON and curve NPZ files with experiment configuration and checkpoint identity.

## 8. Verification status

The evaluator and plotter can be tested without local model weights using synthetic NPZ fixtures. Full numerical paper results still require completed checkpoints, full test negatives, packet-level artifacts, and hardware measurements. The source PDFs were text-extracted for scope verification; Poppler was unavailable in the current Windows environment, so PDF page rendering was not used during this code pass.
