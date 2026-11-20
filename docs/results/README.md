# Released aggregate results

These files are compact, reviewable aggregates from the completed DreamMem experiments. Raw denoising trajectories, model checkpoints, and the larger per-episode result archive are intentionally not bundled.

- `method_metrics.csv`: seed-level means and seed standard deviations by task, split, and method.
- `ablation_metrics.csv`: the corresponding 17-method ablation aggregates.
- `compute_costs.csv`: retained candidates, generated neighbors, controller evaluations, evidence evaluations, timing, and backbone forward-pass accounting.
- `paired_effects.csv`: paired episode-bootstrap effects; episodes are resampled within each seed and seed estimates are equally weighted.
- `paper_tables.md`: automatically rendered main, extended, decoder, and full-ablation tables.
- `metadata.json`: final run count, primary row count, and bootstrap configuration.
- `validation.json` and `inventory.csv`: the final 18-run machine audit.

The statistics describe the disclosed synthetic protocols. Single-seed standard deviations are not estimated, and query accuracy is not a substitute for exact concept recovery. See [the paper](../../PAPER.md) and [the experimental manual](../EXPERIMENTS.md) for method definitions, negative results, and limitations.
