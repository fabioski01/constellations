# Connected propagation on a 36 by 18 target grid

The first grid campaign leaves the single-case force model, initialization,
planner and impulse law unchanged. It evaluates one E2 configuration:
700 km initial mean altitude, 50 degree inclination, +50 km drift offset,
15/1 regional repeat ratio, and 0.25 degree longitude tolerance. The request
epoch is 2025-05-05 12:00 UTC, with initial mean node and phase zero.

The 648 requests comprise 36 periodic longitudes from -180 to +170 degrees
and 18 latitudes including both +/-49 degree endpoints. This coarse grid
was calculated independently; it is not a reconstruction or validation of
the unavailable 360x180 per-target archive. No correction, added force,
empirical margin or relaxed acceptance criterion has been introduced.

## Reproduce and resume

```bash
python -m pip install -r requirements-paper1-validation.txt
python -m unittest discover -s tests -v
python paper1_validate_grid.py \
  --config configs/paper1_validation_36x18.json \
  --output-dir results/paper1/validation-36x18 --workers 8
```

Each completed target is written atomically under `cases/`, and the parent
process records its SHA256 in `checkpoint.jsonl`. An interruption can be
continued with `--resume` only if the configuration, numerical source hashes
and retained checkpoint bytes match. Numerical failures are retained and cause
an unsuccessful exit. They cannot disappear from requested denominators.

The saved `case_evidence.jsonl.gz` contains the complete records, ordered
by target index, including every measured latitude crossing, selected nominal
solution, convergence comparison, repeat diagnostic and pre/postburn state.
The complete raw archive is distributed separately and is available from the
corresponding author. The repository retains the readable per-maneuver CSV,
summary statistics, source code and vector figures. Place the accompanying
archive in the results directory before regenerating the report. To recover
individual checkpoints from that archive:

```bash
python - <<'PY'
import gzip, json
from pathlib import Path
folder = Path('results/paper1/validation-36x18')
(folder / 'cases').mkdir(exist_ok=True)
with gzip.open(folder / 'case_evidence.jsonl.gz', 'rt', encoding='utf-8') as handle:
    for line in handle:
        item = json.loads(line)
        index = item['row']['case_index']
        # The archive retains the exact canonical JSON line used for each case.
        (folder / 'cases' / f'case_{index:04d}.json').write_text(line, encoding='utf-8')
PY
python paper1_validate_grid.py --resume --workers 8
```

Regenerate only statistics and figures, without propagating trajectories:

```bash
python paper1_validation_report.py --data-dir results/paper1/validation-36x18
python paper1_validation_example_figure.py --output-dir results/paper1/validation-36x18
```

The second command reproduces the unchanged Los Angeles illustration in a
publication layout; Los Angeles is separate from the 648 grid targets.

## Physical comparison and event definitions

The scalar corrected planner supplies each target's waiting and drift times.
One shared initial Cartesian state is matched to the stated circular mean
orbit using the existing symmetric one-orbit numerical averaging convention.
It is independent of the requested targets and burns, but is not an exact
theory-specific Brouwer-Lyddane or SGP4 transformation. Its sensitivity remains
an explicit limitation.

Cartesian central gravity plus J2 dynamics are integrated with DOP853. Fixed
signed Hohmann magnitudes are applied in the actual instantaneous RTN
transverse direction at the nominal burn times. The position remains continuous
and the achieved state starts the next arc. Intermediate or terminal ideal
orbits are never imposed. Surface coordinates follow the original validation's
radial WGS84 intersection and shared continuous GMST convention.

Every trajectory is integrated twice, from identical initial Cartesian states,
using the original and ten-times-tighter tolerances and a maximum step of 60
and 30 seconds, respectively. All 648 comparisons must satisfy the 0.1 m
sampled position and 0.001 s event-time gates, identical event counts, and
unchanged matched-event and extended-window acceptance classifications.

Three acceptance quantities are separate:

1. **Longitude only:** the corresponding latitude crossing lies inside the
   original longitude band, regardless of insertion eligibility.
2. **Full matched-event acceptance:** that crossing also occurs at or after
   regional insertion. The matched event is the nearest event of the nominal
   ascending/descending branch within +/- half a nominal regional nodal period.
   Selection is independent of whether the event is successful. All 648 events
   are matched in this run. Predicted access equals final insertion in all 648
   cases, so negative crossing-time errors fail insertion eligibility.
3. **Extended-window acceptance:** some crossing of either branch satisfies the
   original band after insertion, before insertion plus 1.15 nominal repeat
   intervals (about 27.15 hours). A later success does not confirm the planned
   response time. Its first occurrence and delay are saved explicitly.

Longitude **prediction error** is wrapped propagated minus nominal longitude
at corresponding latitude crossings. Longitude **target offset** is wrapped
propagated minus target longitude. The latter is tested against the tolerance;
they must not be interchanged. Fixed-epoch ground prediction distance is a
separate spherical diagnostic, not a circular acceptance footprint.

## Measured results

All 648 requests have nominal solutions, corresponding propagated events and
successful numerical convergence checks. There are no unresolved numerical
cases. Longitude-only acceptance is 385/648 (59.4136%); full matched-event
acceptance is 15/648 (2.3148%). There are 563 early crossings. The disjoint
categories are 15 accepted, 370 before insertion only, 70 longitude only, and
193 failing both conditions. The longer post-insertion window contains an
accepted crossing for 439 requests (67.7469%), including 424 without a
successful matched event. These delayed accesses do not validate nominal time.
Among extended-window successes, the median delay beyond prediction is
23.6127 h, approximately one nominal repeat interval.

The longitude prediction-error MAE is 0.0301258 degrees, RMS 0.0477856 degrees,
95th absolute percentile 0.1311915 degrees, and maximum absolute error
0.1381700 degrees. Signed bias is -0.00547985 degrees. The matched crossing-time
MAE is 0.495022 s, with a -2.289903 to +1.153194 s range. The mean longitude
target-offset magnitude is 0.2306503 degrees; its maximum is 0.3756663 degrees.

The mean fixed-epoch ground prediction discrepancy is 2.80505 km. Mean 15-node
closure is 0.555512 km (maximum 1.269782 km); the mean absolute repeat-time
error is 0.657858 s. Terminal numerical-mean semimajor-axis MAE is 35.8382 m.
These numerical means use the declared averaging convention.

The largest primary-versus-tighter sampled position difference is 0.000123432 m
(0.123432 mm); the largest latitude-event time difference is 1.61236e-8 s.
The 34-test suite passes, and the grid worker reproduces the previously saved
Los Angeles numerical evidence exactly. `verification.json` records checks of
source/output hashes and an independent recount from archived event values.

The raw maneuver therefore does not reliably preserve the strict nominal
access condition for this configuration. The result is evidence of a measured
limitation, not a passed feasibility claim. First-order averaging, initial-state
conversion and ideal burn execution have not been isolated as causal sources.
597 nominal requests lie within 1e-6 degree of the longitude-band boundary,
which explains the sensitivity of acceptance to small discrepancies.

## Saved outputs

- `per_maneuver.csv`: one row per request, including timing, error quantities,
  separate acceptance flags, failure category, terminal/repeat diagnostics,
  cost and numerical convergence results.
- `summary.json`, `results.md`: aggregate statistics and their definitions.
- `grouped_summary.csv`: latitude, branch and duration-quartile diagnostics.
- `manifest.json`, `completion.json`, `checkpoint.jsonl`, `checksums.json`:
  exact inputs, initialization, source versions, completion and saved-byte hashes.
- `case_evidence.jsonl.gz`: separately distributed full event and burn evidence for every target.
- `validation_grid.svg`, `validation_diagnostics.svg` (PNG generated locally): scientific
  overview, timing/longitude feasibility and achieved repeat closure.
- `los_angeles_validation.svg` (PNG generated locally): separate worked-example publication figure.

Primary percentages weight coordinate points equally. Separate spherical
cell-weighted percentages are included; latitude cells are clipped at +/-49
degrees. Neither measure represents target demand. The experiment is one
configuration and one request state on a deterministic coarse grid; no
random-sample confidence interval or general campaign success rate is implied.
