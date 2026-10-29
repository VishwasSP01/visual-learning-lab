# Frozen instruction-editor test evaluation

## Protocol committed before the first model-performance test

The [frozen plan](experiments/semantic_dilated_v2_frozen_plan.json) binds the
existing seed-42 **SemanticDilatedEditor** checkpoint to the instruction manifest
and the previous **InstructionEditor** checkpoint. The semantic model uses
three-class argmax with lowest-index tie breaking and fixed categorical class
selection. The reference keeps its validation-selected **0.95 cutoff**.

Only these two existing models are evaluated. There is no training, alternate
seed comparison, threshold search, decoder selection, or other setting change.
The primary metric is exact scene pairs: both full targets must match. Report
exact requests, each instruction separately, requested-pixel accuracy,
opposite-object/background preservation, error counts, and paired changes.

The dataset reserves **500 test scenes / 1,000 requests**, with the same two
requests grouped by scene. Manifest-only checks precede image loading. Both
checkpoint hashes and the manifest hash must match before any test image is
opened. A dedicated frozen evaluator loads test images; the training loader's
test prohibition stays in place. Final inference and scoring use CPU.

```bash
uv run --locked python scripts/run_experiment.py --config configs/semantic_dilated_v2_frozen_test.json
```

The output directory must be new. An evaluation-start marker prevents silent
reuse after an interrupted run. The command saves reports, per-request cases,
binary predicted masks, and the first four test requests in manifest order.
Saved masks allow an independent pixel-count audit without another model pass.
The report fingerprints models, plan, manifest, evaluated files, saved masks,
code, dependencies, and Git revision. Model/plan bytes are checked again at end.

Construction integrity checks had already read test files. The relevant claim
is that **test model performance did not guide development or selection**.
After this run the test is observed; repeated evaluations of the same test
are reproduction checks rather than fresh evidence. Future tuning must not be
presented as retaining an unseen test on these same examples.

These are synthetic shapes from one rendering process with held-out circle
positions. They do not establish real-photo editing or language understanding.
