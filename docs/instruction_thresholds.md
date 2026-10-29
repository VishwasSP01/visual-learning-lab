# Instruction-model validation thresholds

## Protocol declared before the full sweep

Keep the 24-epoch conditioned checkpoint fixed. Verify its SHA-256 and the
manifest against the original validation report, then reproduce all original
0.50 validation scores before accepting a comparison. Infer once on CPU using
RGB and the requested categorical ID; reuse those logits for every candidate.

Compare the fixed shared grid **0.50, 0.70, 0.80, 0.90, 0.95**. Use one cutoff
for both instructions. Select the most exact **scene pairs** (both requests
completely correct), then most exact requests, then fewest wrong pixels, then
lowest threshold. No per-instruction cutoff search or retraining is part of
this experiment. The new test images remain reserved.

Report per-instruction missed requested pixels, unwanted opposite-object edits,
and unwanted background edits. Compare improvements and regressions on each
request and scene against 0.50. Preview both requests for the two worst scenes
at 0.50, breaking ties by scene ID; this is an explicitly selected failure
preview, not a representative sample.

```bash
uv run --locked python scripts/run_experiment.py --config configs/instruction_cnn_v2_thresholds.json
```

The output directory must be new. The command saves the report, all candidate
case records, a selected configuration, and a before/after preview. It refuses
changed reference files and preserves existing output. This is validation-based
operating-point selection; sigmoid scores are not proven correctness
probabilities. Improvements here will require a later frozen test evaluation.
