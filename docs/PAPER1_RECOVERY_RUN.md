# Run the dense study in complete stages

`paper1_dense_recovery_run.py` uses the unchanged numerical sources from the
dense-study implementation. It finishes the corrected grids before starting
the slower original-code comparisons. Each chunk has an atomic receipt with
its checksum; the JSONL index is replaced atomically rather than held open
throughout a long run. Resume validates the configuration and all numerical
source hashes before reusing a chunk.

From the repository root:

```sh
python -m unittest discover -s tests -v
python paper1_dense_recovery_checks.py --output-dir ../dense-results/checks --workers 6
python paper1_dense_recovery_run.py --stage corrected --output-dir ../dense-results/corrected --workers 8
python paper1_dense_audit.py --data-dir ../dense-results/corrected --output-dir ../dense-results/corrected/audit
python paper1_dense_recovery_figures.py
python paper1_dense_recovery_run.py --stage paired --output-dir ../dense-results/paired --workers 8
python paper1_dense_audit.py --data-dir ../dense-results/paired --output-dir ../dense-results/paired/audit
python paper1_dense_recovery_comparison.py
```

Use the same output directory with `--resume` to continue. The corrected stage
contains the common-domain altitude choices, six-band comparisons at +50 km
and E2 ±200 km, the representative coarse/medium/shifted grids and native-domain
diagnostics. The paired stage compares the original and corrected sources at
+50 km in all five cases and ±200 km in E2, at all six bands. Each stage has
its own explicitly bounded manifest and completion flag. Neither stage
includes the earlier full configuration's additional 10 km altitude sweep
or historical-caption bridge profiles.

The checks compare the batched sources with scalar calculations, test the
original clock at rounding boundaries and reconstruct the supplied six-target
CSV across its altitude offsets. The independent audit checks the nominal
secular geometry. These checks do not integrate executed Cartesian maneuvers.

Figures use `reconfig_plot_from_csv.py` and
`reconfig_plotter_multicore.py`. The saved-data path needs no Cartopy download.
Numerical data and manuscript files are distributed separately from these
source changes.

The comparison exporter also requires the completed corrected stage. It
checks that the repeated corrected arrays agree bitwise, then uses the
existing component plotter for original/corrected tolerance curves. Figures
are written atomically; the CSV plotter uses regular minor-tick spacing even
when the exact regional radius adds an irregular major tick.
