# Plot Directory File and Input Reference

## 1. Scope

This document explains every file under `plot/`, the figure produced by each plotting entry point, and the files that must exist before a figure can be generated. Run all commands from the repository root.

The most important distinction is between two kinds of evaluation files:

- `vista_augur/run_train.py --evaluate` writes a raw `evaluation_summary.json` and an `evaluation_scores.npz` file.
- The plotting programs expect the standardized `*.summary.json` and, for ROC figures, the paired `*.curves.npz` written by `python -m data.evaluate`.

Therefore, `evaluation_summary.json` is not a drop-in replacement for `*.summary.json`. The normal path is:

```text
trained checkpoint
    |
    +-- training ----------------------> train_summary.json
    |
    +-- run_train.py --evaluate ------> evaluation_scores.npz
                                             |
                                             +-- data.evaluate --> *.summary.json
                                                                  *.curves.npz
                                                                  index.json
```

Additional packet-level arrays, square correlation matrices, and timing measurements are separate inputs described in Section 4.

## 2. Runtime prerequisites

The plot programs require Python, NumPy, and Matplotlib. The repository requirements already declare compatible versions:

```powershell
python -m pip install -r requirements.txt
```

The plot programs use Matplotlib's non-interactive `Agg` backend. They do not open a desktop window. The requested figure is written directly to disk, and its parent directory is created automatically. PDF and SVG outputs use a transparent background; other suffixes, such as PNG, use the normal background.

The commands must be run from the repository root so that imports such as `plot.common` resolve correctly:

```powershell
python -m plot.plot_roc ...
```

## 3. File-by-file inventory

### 3.1 `plot/__init__.py`

This file marks `plot` as a Python package and describes the package as the publication-plotting utilities for Work2. It does not read data or generate a figure.

Required input files: none.

### 3.2 `plot/common.py`

This is a shared helper module, not a command-line plotting entry point. It provides four behaviors used by the other files:

1. It selects the Matplotlib `Agg` backend and defines the common font sizes, DPI, legend sizes, and removal of the top and right axes spines.
2. It reads standardized summary JSON files.
3. It locates the curve NPZ relative to the summary JSON. If the summary contains `"curves": "run.curves.npz"`, the loader expects `run.curves.npz` in the same directory as the summary.
4. It provides shared figure saving, condition labels, and colors. A condition named `clean` is displayed as `No-Perturbation`; adversarial-like conditions use the summary's `method` name.

Required input files: none by itself. The schemas consumed through these helpers are documented in Section 4.

### 3.3 `plot/plot_roc.py`

Purpose: draw the main attack-effectiveness ROC figure from one or more standardized experiment summaries.

Figure structure:

- One horizontal panel is created for each distinct `target` value found in the supplied summaries.
- The x-axis is false positive rate (FPR) on a logarithmic scale.
- The y-axis is true positive rate (TPR).
- Every condition in every summary contributes one ROC curve to the panel for that target.
- A dashed diagonal reference line represents random ranking.
- `--min-fpr` controls both the left x-axis limit and the start of the diagonal reference line. It must be finite and greater than zero.
- `--title` optionally adds a figure-level title.

Interpretation: for the same frozen correlation attacker and test protocol, a lower defended TPR at a fixed FPR indicates stronger suppression by the defense. The clean curve should remain high enough to establish that the attacker and test data are functioning.

Direct prerequisites:

- One or more standardized `*.summary.json` files.
- The `*.curves.npz` named by the `curves` field of each summary. It must be in the same directory as that summary unless the `curves` value itself contains a relative subpath.

Default output: `plot/results/roc.pdf`.

Example:

```powershell
$summaries = Get-ChildItem 'data/results/*.summary.json' | ForEach-Object FullName
python -m plot.plot_roc $summaries --min-fpr 0.0001 --output plot/results/main_roc.pdf
```

### 3.4 `plot/plot_operating_points.py`

Purpose: compare precision, recall, and F1 at a selected low-FPR operating point.

Figure structure:

- Each x-axis group represents one `(target, condition)` pair.
- The group label contains the target name on the first line and the display condition on the second line.
- Each group has three bars: precision, recall, and F1.
- The y-axis is a score from 0 to 1.
- `--fpr` selects an exact key from `conditions[*].operating_points`; the default is `fpr_1e-04`.
- Conditions that do not contain the selected key are skipped. The program fails if no supplied condition contains it.

Interpretation: lower defended recall or F1 at the same low-FPR protocol indicates reduced attacker effectiveness. These bars use the condition-specific, recalibrated threshold stored under `operating_points`; they do not use `fixed_clean_threshold`.

Direct prerequisite:

- One or more standardized `*.summary.json` files containing the requested operating-point key. The curve NPZ is not read by this program.

Default output: `plot/results/operating_points.pdf`.

Example:

```powershell
$summaries = Get-ChildItem 'data/results/*.summary.json' | ForEach-Object FullName
python -m plot.plot_operating_points $summaries --fpr fpr_1e-04 --output plot/results/f1_1e-4.pdf
```

### 3.5 `plot/plot_score_matrix.py`

Purpose: visualize one or more square ingress-to-exit correlation-score matrices as heat maps.

Figure structure:

- One horizontal panel is created for each NPZ key supplied with `--keys`.
- Matrix rows are labeled `Ingress flow`; columns are labeled `Exit flow`.
- Cell color is the attack model's correlation score.
- All panels share one global color range computed from the minimum and maximum across all selected matrices. This makes clean and defended panels visually comparable.
- `--max-items` keeps the leading top-left square of each matrix; the default is 100 flows.
- `--labels` optionally provides panel titles. If omitted, titles are derived from the NPZ keys. The label count must equal the key count.
- Every selected array must be two-dimensional and square after slicing.

Interpretation: the diagonal represents scores for true ingress/exit pairs when row and column order is aligned; off-diagonal cells represent mismatched pairs. Effective correlation suppression normally reduces separation between the diagonal and off-diagonal scores. This interpretation is valid only if the producer preserved the same session ordering on both axes.

Direct prerequisite:

- One NPZ file containing one or more square matrices under known keys, for example `clean_matrix` and `adv_matrix`.

Default output: `plot/results/score_matrices.pdf`.

Example:

```powershell
python -m plot.plot_score_matrix `
  --input data/matrices/deepcorr.npz `
  --keys clean_matrix adv_matrix `
  --labels Clean Work2 `
  --max-items 100 `
  --output plot/results/deepcorr_matrix.pdf
```

Current availability: the generic Work2 trainer does not write a clean/defended square matrix. A compatible matrix must come from a target-model evaluation/export step or a legacy DeepCoFFEA `corr_matrix` artifact. Do not mistake `evaluation_scores.npz`, which contains one-dimensional positive and negative score arrays, for a score-matrix file.

### 3.6 `plot/plot_overhead.py`

Purpose: compare normalized perturbation cost across targets or experiment summaries.

Figure structure:

- Each x-axis group represents one summary and is labeled by its `target` value, falling back to `name`.
- Four bars are drawn per group:
  - `Time L2`: relative L2 norm of the time-channel perturbation;
  - `Size L2`: relative L2 norm of the size-channel perturbation;
  - `Delay overhead`: total positive time addition divided by total original absolute time magnitude;
  - `Size overhead`: total positive size addition divided by total original absolute size magnitude.
- The y-axis is `Relative overhead` and is not clamped to 0-1.
- Summaries without an `overhead` object are skipped. The program fails if none contains overhead data.

Direct prerequisite:

- One or more standardized `*.summary.json` files whose experiment records were evaluated with an `overhead_artifact`.
- Each such summary should contain both `target` and `name`. The current fallback expression evaluates `name` even when `target` exists, while summaries produced by `data.evaluate` always provide both fields.

Default output: `plot/results/overhead.pdf`.

Example:

```powershell
$summaries = Get-ChildItem 'data/results/*.summary.json' | ForEach-Object FullName
python -m plot.plot_overhead $summaries --output plot/results/overhead.pdf
```

Current availability: the trainer computes normalized L2 terms internally but does not currently export the packet-level `original`, `adversarial`, and optional `valid_mask` arrays required by `data.evaluate` to construct the full `overhead` object. Those arrays must be exported separately before this figure can be produced from a new run.

### 3.7 `plot/plot_training.py`

Purpose: show optimization and evaluation dynamics over training epochs.

Figure structure:

- Panel 1, `Objective`: evaluation loss versus epoch.
- Panel 2, `Attack response`: adversarial positive rate versus epoch. This is the fraction of true matched flows still classified as positive after perturbation; lower is better for the defense, subject to a valid unchanged evaluation protocol.
- Panel 3, `Perturbation cost`: time relative L2 as a solid line and size relative L2 as a dashed line.
- Supplying multiple summaries overlays multiple settings. The legend label is `setting_name`, or the summary's parent directory name when `setting_name` is absent.

Direct prerequisite:

- One or more training-generated `train_summary.json` files.

Default output: `plot/results/training_dynamics.pdf`.

Example:

```powershell
$trainingSummaries = Get-ChildItem 'vista_augur/outputs/checkpoints/*/train_summary.json' | ForEach-Object FullName
python -m plot.plot_training $trainingSummaries --output plot/results/training.pdf
```

Current limitation: Work1-aligned DeepCoFFEA training records `eval: null` for each epoch because it monitors training metrics without running the evaluation loader. `plot_training.py` directly indexes `item["eval"]`, so such a DeepCoFFEA `train_summary.json` is not currently compatible with this plotter. DeepCorr and m-DeepCorr summaries include epoch evaluation records and match the expected schema.

### 3.8 `plot/plot_efficiency.py`

Purpose: compare the distribution summary of a selected efficiency metric across reports.

Figure structure:

- Each x-axis position is one report, labeled by the report filename stem.
- The bar height is the metric mean.
- A diamond marks P95.
- An `x` marks P99.
- `--metric` selects the top-level report key. The default is `latency_ms`.
- Reports that lack the selected metric are skipped. The program fails if no supplied report contains it.

Common metric keys produced by `data.benchmark` and `data.evaluate_efficiency` are:

- `latency_ms`;
- `throughput_per_second`;
- `cpu_percent`;
- `peak_ram_mb`.

Direct prerequisite:

- One or more efficiency JSON reports containing `mean`, `p95`, and `p99` under the selected metric key.

Default output: `plot/results/efficiency.pdf`.

Example:

```powershell
python -m plot.plot_efficiency `
  data/results/deepcorr_efficiency.json `
  data/results/mdeepcorr_efficiency.json `
  --metric latency_ms `
  --output plot/results/latency.pdf
```

### 3.9 `plot/plot_ablation.py`

Purpose: draw component or mechanism ablation ROC curves.

This is a thin wrapper around `plot_roc.py`. It produces the same panel layout, logarithmic FPR axis, TPR axis, condition curves, and random-reference diagonal, but fixes the figure title to `Ablation study` and the minimum FPR to the ROC helper's default of `1e-4`.

The meaning of each line comes entirely from the supplied experiment summaries. Valid examples include time-only, size-only, both-channel, no-prompt, no-temporal-branch, no-visual-branch, or alternative-reprogramming experiments. Each genuine model-component ablation should normally come from its own trained checkpoint; renaming a condition does not create an ablation.

Direct prerequisites:

- One or more standardized ablation `*.summary.json` files.
- The paired curve NPZ named by each summary.

Default output: `plot/results/ablation.pdf`.

Example:

```powershell
$ablationSummaries = Get-ChildItem 'data/results/ablation/*.summary.json' | ForEach-Object FullName
python -m plot.plot_ablation $ablationSummaries --output plot/results/ablation.pdf
```

### 3.10 `plot/plot_transferability.py`

Purpose: draw white-box, black-box transfer, and/or adaptive-attacker ROC comparisons.

This is also a thin wrapper around `plot_roc.py`. It uses the same ROC layout and data contract. Its default title is `Transferability and adaptive attack`, which can be changed with `--title`. The minimum FPR is fixed to the ROC helper's default of `1e-4`.

The plotter does not infer whether a run is genuinely white-box, black-box, or adaptive. That experimental meaning must be established by the checkpoint/training protocol and encoded clearly in the summary's `method` and condition names.

Direct prerequisites:

- One or more standardized transfer/adaptive `*.summary.json` files.
- The paired curve NPZ named by each summary.

Default output: `plot/results/transferability.pdf`.

Example:

```powershell
$transferSummaries = Get-ChildItem 'data/results/transfer/*.summary.json' | ForEach-Object FullName
python -m plot.plot_transferability $transferSummaries --output plot/results/transferability.pdf
```

### 3.11 `plot/plot_all.py`

Purpose: generate a configured set of figures from one JSON figure manifest.

This file does not combine figures. It reads each record in sequence and launches the corresponding `python -m plot.<module>` command as a subprocess. Supported `type` values are:

```text
roc
operating_points
score_matrix
overhead
training
efficiency
ablation
transferability
```

Manifest behavior:

- `base_dir` is resolved relative to the figure manifest's directory. If omitted, it defaults to that directory.
- Every string in `inputs` becomes a positional argument and is resolved against `base_dir`.
- Every `options` key becomes a CLI option by changing underscores to hyphens; for example, `min_fpr` becomes `--min-fpr`.
- Option values may be scalars or lists. Lists are required for options such as `keys` and `labels`.
- Only option keys named `input` and `output` are resolved as paths against `base_dir`. Other option values are passed literally.
- Processing stops on the first failed figure command.

Direct prerequisite:

- One figure-manifest JSON file plus every data file referenced by that manifest.

Example:

```powershell
python -m plot.plot_all --manifest plot/figures.example.json
```

`plot/figures.example.json` uses `"base_dir": ".."`, which resolves paths from the repository root because the manifest itself is inside `plot/`.

### 3.12 `plot/figures.example.json`

This is an example manifest for `plot_all.py`, not a data result and not a plotting program. It currently defines two figures:

1. `main ROC figure`, reading `data/results/work2_deepcorr.summary.json` and writing `plot/results/main_roc.pdf`;
2. `low-FPR metrics`, reading the same summary and writing `plot/results/operating_points.pdf` at `fpr_1e-04`.

The paths are placeholders until the named standardized summary and its curve NPZ have been generated. Copy this file to a new experiment-specific manifest rather than treating the example paths as guaranteed repository artifacts.

## 4. Required file schemas and how to obtain them

### 4.1 Standardized evaluation summary: `*.summary.json`

Used directly by:

- `plot_roc.py`;
- `plot_operating_points.py`;
- `plot_overhead.py`;
- `plot_ablation.py`;
- `plot_transferability.py`.

Minimum fields vary by plot. A complete representative structure is:

```json
{
  "schema_version": 1,
  "name": "work2_deepcorr",
  "target": "DeepCorr",
  "method": "Work2",
  "category": "work2",
  "source": "absolute/or/auditable/source/path",
  "curves": "work2_deepcorr.curves.npz",
  "conditions": {
    "clean": {
      "positive_count": 1000,
      "negative_count": 199000,
      "roc_auc": 0.99,
      "average_precision": 0.95,
      "operating_points": {
        "fpr_1e-04": {
          "target_fpr": 0.0001,
          "actual_fpr": 0.0001,
          "threshold": 0.8,
          "negative_count": 199000,
          "minimum_resolvable_fpr": 0.0000050251,
          "resolvable": true,
          "precision": 0.9,
          "recall": 0.8,
          "f1": 0.8471,
          "fpr": 0.0001,
          "tpr": 0.8,
          "tp": 800,
          "fp": 20,
          "tn": 198980,
          "fn": 200
        }
      }
    },
    "adv": {
      "operating_points": {
        "fpr_1e-04": {
          "precision": 0.5,
          "recall": 0.1,
          "f1": 0.1667
        }
      }
    }
  },
  "overhead": {
    "relative_l2": {"time": 0.1, "size": 0.05},
    "aggregate": {
      "relative_delay_overhead": 0.2,
      "relative_size_overhead": 0.08
    }
  }
}
```

Do not manually fabricate paper metrics. Generate this file with `data.evaluate`, then validate it with `data.validate_results`.

### 4.2 ROC curve bundle: `*.curves.npz`

Used by `plot_roc.py` and, through it, the ablation and transferability wrappers.

For every condition key in the summary, at least these arrays must exist:

```text
<condition>__fpr
<condition>__tpr
```

For example:

```text
clean__fpr
clean__tpr
adv__fpr
adv__tpr
```

`data.evaluate` also writes `<condition>__thresholds`, `<condition>__precision`, and `<condition>__recall`, although the current ROC plotter does not read them. The curve NPZ is generated beside the summary, and the summary's `curves` field stores its filename.

### 4.3 Raw score artifact: `evaluation_scores.npz`

This is an upstream input to `data.evaluate`, not a direct plot input. A Work2 evaluation writes:

```text
clean_positive_scores
clean_negative_scores
adv_positive_scores
adv_negative_scores
```

Each item is a one-dimensional score array after loading/flattening. The offline evaluator converts matching `*_positive_scores` and `*_negative_scores` pairs into named conditions.

Generate it from a trained checkpoint with the appropriate target configuration. The exact checkpoint and configuration are experiment-specific; a typical invocation is:

```powershell
python vista_augur/run_train.py `
  --config vista_augur/configs/deepcorr_config.jsonc `
  --evaluate `
  --eval-split test
```

The run directory receives `evaluation_scores.npz` and a raw `evaluation_summary.json`. Use the score NPZ in a data-evaluation manifest.

### 4.4 Data-evaluation manifest: for `data.evaluate`

The repository example is `data/manifest.example.json`. Its `artifact` path is only a placeholder and must be replaced. A useful full record is:

```json
{
  "base_dir": "..",
  "experiments": [
    {
      "name": "work2_deepcorr",
      "target": "DeepCorr",
      "method": "Work2",
      "category": "work2",
      "artifact": "vista_augur/outputs/checkpoints/RUN/evaluation_scores.npz",
      "fpr_targets": [0.001, 0.0001],
      "bootstrap": 0
    }
  ]
}
```

Generate and validate the standardized outputs:

```powershell
python -m data.evaluate --manifest data/my_experiments.json --output-dir data/results
$summaries = Get-ChildItem 'data/results/*.summary.json' | ForEach-Object FullName
python -m data.validate_results $summaries
```

At least 10,000 negative scores are required to resolve an empirical FPR of `1e-4`; the intended DeepCorr test protocol uses more. Validation rejects unresolvable operating points by default.

### 4.5 Packet-level overhead artifact

This is an upstream NPZ used by `data.evaluate` when an experiment manifest contains `overhead_artifact`. It must contain:

```text
original       # required, shape (..., channels, length)
adversarial    # required, same shape as original
valid_mask     # optional, excludes padding
```

The experiment record must also identify time and size channels:

```json
{
  "overhead_artifact": "data/artifacts/deepcorr_overhead.npz",
  "original_key": "original",
  "adversarial_key": "adversarial",
  "mask_key": "valid_mask",
  "time_channels": [0],
  "size_channels": [1],
  "time_unit_to_ms": 1.0,
  "size_unit_to_kb": 1.0
}
```

`data.evaluate` then inserts the computed `overhead` object into the standardized summary. The current training/evaluation path does not save these arrays automatically, so a separate packet-level export is presently required.

### 4.6 Square score-matrix NPZ

This file is read directly by `plot_score_matrix.py`. Every requested key must map to a square two-dimensional numerical array:

```text
clean_matrix -> shape (N, N)
adv_matrix   -> shape (N, N)
```

The arrays must use identical row and column ordering if the diagonal is to represent matched pairs. A legacy DeepCoFFEA `corr_matrix` may be plotted directly if it is square, but its window-block layout should be understood before interpreting its diagonal or truncating it with `--max-items`.

There is no generic matrix exporter in the current Work2 trainer. This file must come from a target-specific evaluator or a compatible legacy artifact.

### 4.7 Training summary: `train_summary.json`

Written automatically at the end of `trainer.train()`. The fields consumed by `plot_training.py` are:

```json
{
  "setting_name": "experiment label",
  "epochs": [
    {
      "epoch": 1,
      "train": {},
      "eval": {
        "loss": 1.0,
        "adv_positive_rate": 0.5,
        "time_ratio": 0.1,
        "size_ratio": 0.05
      }
    }
  ]
}
```

The plotter reads only `epoch` and the four shown `eval` metrics; it does not plot the `train` object.

### 4.8 Efficiency measurement NPZ and report JSON

`data.benchmark` produces a measurement NPZ containing repeated samples, normally:

```text
latency_ms
throughput_per_second
cpu_percent
peak_ram_mb
```

Example benchmark command:

```powershell
python -m data.benchmark `
  --factory my_benchmark:build_generator_call `
  --factory-kwargs '{"config":"vista_augur/configs/deepcorr_config.jsonc"}' `
  --warmup 10 `
  --repetitions 100 `
  --items-per-call 16 `
  --output data/measurements/deepcorr.npz
```

The referenced callable or factory is experiment-specific; `my_benchmark:build_generator_call` is an example interface, not a module currently supplied by this repository.

Convert the measurement arrays into the JSON consumed by `plot_efficiency.py`:

```powershell
python -m data.evaluate_efficiency `
  --input data/measurements/deepcorr.npz `
  --output data/results/deepcorr_efficiency.json
```

Every metric in the resulting JSON has this schema:

```json
{
  "latency_ms": {
    "count": 100,
    "mean": 12.3,
    "std": 1.2,
    "median": 12.1,
    "p90": 13.5,
    "p95": 14.0,
    "p99": 15.2,
    "min": 10.1,
    "max": 16.0
  }
}
```

The plotter currently uses only `mean`, `p95`, and `p99`.

## 5. Minimum prerequisites by desired figure

| Desired figure | Files that must already exist | How they are normally obtained |
| --- | --- | --- |
| Main ROC | `*.summary.json` plus its named `*.curves.npz` | `evaluation_scores.npz` -> `data.evaluate` |
| Low-FPR precision/recall/F1 | `*.summary.json` with the requested operating point | `evaluation_scores.npz` -> `data.evaluate` |
| Score matrices | One NPZ containing selected square matrix keys | Target-specific or legacy matrix export; not produced by the generic trainer |
| Relative overhead | `*.summary.json` containing `overhead` | Packet-level overhead NPZ -> `data.evaluate` |
| Training dynamics | `train_summary.json` with non-null epoch `eval` records | Produced by training for DeepCorr/m-DeepCorr |
| Efficiency | Efficiency report JSON | `data.benchmark` NPZ -> `data.evaluate_efficiency` |
| Ablation ROC | Ablation `*.summary.json` files and their curve NPZ files | Separately trained/evaluated ablation runs -> `data.evaluate` |
| Transfer/adaptive ROC | Transfer/adaptive `*.summary.json` files and their curve NPZ files | Correctly designed transfer/adaptive runs -> `data.evaluate` |
| All configured figures | Figure manifest plus the union of all referenced inputs above | Copy and edit `plot/figures.example.json` |

## 6. Files currently present in this checkout

The repository currently contains a synthetic QA/smoke bundle under `vista_augur/outputs/qa/`. It is useful for checking that the plotting programs run, but it is not a formal paper result.

Available smoke inputs include:

- `vista_augur/outputs/qa/results/work2_smoke_deepcorr.summary.json` and its paired `work2_smoke_deepcorr.curves.npz`: directly usable by the ROC plotter;
- the same smoke summary contains an `overhead` object and is directly usable by the overhead plotter;
- the smoke summary contains only `fpr_1e-01` and `fpr_1e-02`, so the operating-point plotter must be called with one of those keys rather than its `fpr_1e-04` default;
- `vista_augur/outputs/qa/benchmark_v2.json`: directly usable by the efficiency plotter;
- `vista_augur/outputs/qa/figures.json`: a working two-figure smoke manifest for `plot_all.py`.

For example:

```powershell
python -m plot.plot_roc `
  vista_augur/outputs/qa/results/work2_smoke_deepcorr.summary.json `
  --min-fpr 0.01 `
  --output vista_augur/outputs/qa/documentation_check_roc.png

python -m plot.plot_operating_points `
  vista_augur/outputs/qa/results/work2_smoke_deepcorr.summary.json `
  --fpr fpr_1e-02 `
  --output vista_augur/outputs/qa/documentation_check_operating_points.png

python -m plot.plot_efficiency `
  vista_augur/outputs/qa/benchmark_v2.json `
  --output vista_augur/outputs/qa/documentation_check_efficiency.png
```

No formal `data/results/`, `data/measurements/`, or `data/matrices/` result set is currently present in the checkout, and no training-run `train_summary.json` is present under `vista_augur/outputs/`. Those formal inputs still need to be generated for paper figures.

## 7. Recommended preparation order

1. Train every main, ablation, transfer, and adaptive setting that the paper will compare. Preserve the generated `train_summary.json` files.
2. Evaluate each relevant checkpoint on the unchanged test protocol to obtain `evaluation_scores.npz`.
3. Prepare one `data.evaluate` manifest containing all runs and generate standardized summaries and curve bundles.
4. Validate every standardized summary before plotting low-FPR results.
5. Separately export packet-level original/defended arrays for overhead and square score matrices for correlation heat maps.
6. Benchmark representative inference calls and summarize each measurement NPZ into an efficiency JSON.
7. Create an experiment-specific figure manifest and run `plot.plot_all`, or invoke individual plot modules while iterating.

## 8. Known failure modes

- Passing raw `evaluation_summary.json` to a standardized-summary plot fails because it has no top-level `conditions` object or `curves` reference.
- Moving a summary without moving its curve NPZ, or leaving a stale `curves` filename, breaks ROC loading.
- Requesting `fpr_1e-04` when the summary contains only `fpr_1e-03` produces no operating-point rows.
- Fewer than 10,000 negative samples cannot empirically resolve `1e-4` FPR, even though a numerical threshold can still be calculated.
- Passing `evaluation_scores.npz` to `plot_score_matrix.py` fails because the score arrays are one-dimensional, not square.
- A hand-written overhead summary that has `target` but omits `name` fails in the current overhead plotter; standardized evaluator output includes both.
- Comparing heat maps produced with different row/column order makes diagonal interpretation invalid.
- A DeepCoFFEA `train_summary.json` with `eval: null` fails in the current training plotter.
- An efficiency JSON without `mean`, `p95`, or `p99` under the selected metric fails when the plotter indexes those fields.
- In `plot_all`, a misspelled figure `type` fails immediately, and any failed child command stops the remaining figure build.

## 9. Output handling

Generated figures default to `plot/results/`, which is separate from the source data. Preserve the following together for reproducibility:

- figure manifest;
- standardized summary JSON files;
- curve NPZ files;
- training summaries;
- matrix and overhead artifacts;
- efficiency reports;
- experiment configurations and checkpoint identities.

The PDF, SVG, or PNG alone is not sufficient evidence for a result because it does not retain the evaluation protocol, sample counts, threshold resolvability, or source checkpoint.
