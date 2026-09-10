# Work2 checkpoint evaluation

These entry points keep the Work1 evaluation protocol unchanged except for the
Work2 generator forward pass and its required two-GPU placement:

```bash
python eval/deepcorr300/evaluate_deepcorr300.py
python eval/mdeepcorr/evaluate_mdeepcorr.py
python eval/deepcoffea/evaluate_deepcoffea.py
```

Place exactly one generator checkpoint (`.pt` or `.pth`) beside the matching
evaluator, or select one with `--checkpoint`.

## Work1 defaults

| Evaluator | Generator batch | Workers | Shuffle | Drop last | Target batch | Evaluated sessions |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| DeepCorr300 | 16 | 0 | no | yes | 256 | 992 of the 1,000-session test split |
| mDeepCorr | 16 | 32 | no | yes | 64 | 992 of the 1,000-session test split |
| DeepCoFFEA | 25 | 0 | yes | yes | 275 windows | 500 |

The defaults are part of the comparison protocol. CLI overrides remain
available only for bounded smoke tests and must not be used for paper results.

## Output location and names

Outputs are written directly beside the evaluator under `eval/`; they are not
written to `vista_augur/outputs/`.

```text
eval/deepcorr300/
  Gdeepcorr_advsamples_time{time:.4f}_size{size:.4f}.p
  Gtest_index300_time0_result.p

eval/mdeepcorr/
  Gmdeepcorr_advsamples_time{time:.4f}_size{size:.4f}.p
  Gmdeepcorr_threshDC100_time0_0.0100.p

eval/deepcoffea/
  corrmatrix_sim{sim:.4f}.npz
  Gcorrmatrix_time{time:.4f}_size{size:.4f}_sim{sim:.4f}.npz
```

The pickle/NPZ schemas and evaluation-result terminal messages follow the
corresponding Work1 scripts. DeepCorr retains Work1's `199 negative + 1
positive` ordering and final incomplete target batch truncation. mDeepCorr
retains its distinct `1 positive + 199 negative` ordering, per-entry negative
pair seed, DC100 threshold `0.01`, and DC700 cascade. DeepCoFFEA exports the
full flattened-window cosine matrices used by Work1.

The generator requires the frozen Qwen backbone and the Work2 trainable model
to occupy separate logical devices. By default, `--gpus 2,3` maps Qwen to
`cuda:0` and the generator plus frozen target to `cuda:1`. This placement is a
generator-architecture exception and does not alter the evaluation protocol.

Example smoke tests:

```bash
python eval/deepcorr300/evaluate_deepcorr300.py --max-samples 16 --negative-pairs 1
python eval/mdeepcorr/evaluate_mdeepcorr.py --max-samples 16 --negative-pairs 1
python eval/deepcoffea/evaluate_deepcoffea.py --max-steps 1
```
