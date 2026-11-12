# Research reports

- `manuscript_comparison.csv`: Table 2 from the supplied manuscript.
- `manuscript_ablation.csv`: Table 3 from the supplied manuscript.
- `legacy_results.txt`: the original `result.txt`, preserved without editing its numbers or merging separate runs.

- `legacy_timings.csv`: timings transcribed from saved output in the original analysis notebook.
- `local_data_profile.json`: inspection of the supplied local archive/extraction, including zero-byte entries; this is not a model evaluation.

These are historical reported results, not outputs of the restructuring checks. The original result log contains multiple runs with different evaluated pixel counts; retain that provenance. New evaluation output belongs in `outputs/logs/` and includes the mask seed, checkpoint, configuration, and metric protocol.
