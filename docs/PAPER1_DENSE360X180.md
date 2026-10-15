# Paper 1: dense nominal study

The primary grid contains **360 longitudes × 180 latitudes = 64,800 requested
coordinates per configuration**. Latitude is geodetic, equally spaced over
±49 degrees including endpoints. Longitude is periodic from −180 to +179
degrees. Script filenames retain the older latitude × longitude convention.

This study keeps the wait/two-transfer/drift maneuver, the five orbit cases,
the 2025-05-05 12:00 UTC request with zero initial node/phase, and separate
200/300 hour wait/drift caps. There is no additional 72 hour cutoff on the
primary response means. Exact regional roots and the existing gravity/radius
conventions are retained.

## Reproduce

Install `requirements-paper1.txt`. From the repository root:

```bash
python paper1_dense_checks.py --output-dir results/paper1/dense-360x180-20261015/checks --workers 6
python paper1_dense_analysis.py --output-dir analysis/paper1-dense-360x180-20261015 --workers 8
python paper1_dense_audit.py --data-dir analysis/paper1-dense-360x180-20261015 --output-dir results/paper1/dense-360x180-20261015
python paper1_dense_figures.py --data-dir analysis/paper1-dense-360x180-20261015 --output-dir results/paper1/dense-360x180-20261015
```

Use `--resume` to continue the numerical run with unchanged configuration and
numerical source hashes. A failed checksum or changed numerical input is an
error; do not bypass it. `--profiles` selects a subset and does not make an
incomplete full-study manifest complete. Final figures require completion.

## Comparisons and counts

- `common`: identical 64,800 coordinates for the five cases and fourteen
  matched drift choices; six tolerances at all +50 km cases and E2 ±200 km.
- `refined_altitude`: additional corrected samples at 10 km spacing, retaining
  the exact regional radius and common target domain.
- `coarse`, `medium`, `shifted`: representative 90×45, 180×90, and shifted
  360×180 sensitivity checks at all six bands. Latitude grids are not nested.
- `native_requested`: intentionally retains ±59 and ±97 degree request lists
  to quantify coordinate/reachability rejection separately from common-domain
  means. A 98 degree inclination is valid; ±97 degree target latitude is not.
- `thesis_grid_bridge`, `thesis_tolerance_grid`: selected caption-sized source
  reconstructions. Complete historical generators/settings are not recovered.
- `historical_csv`: the supplied six-target E1 sweep at all 41 offsets. This
  source artifact, unlike every thesis figure, has been reproduced exactly.

The byte-preserved original multicore solver is under
`reference/paper1_original`. `paper1_dense_solver.py` batches original scan and
bisection decisions across longitudes, preserving its integer-second clock,
fallback/branch selection, and equal-radius transfer behavior. Separate
scalar comparisons and the historical CSV check validate the baseline.

The corrected calculation derives feasible coast intervals and inverts the
continuous longitude-band entry directly where monotonicity is checked. It
shares target-independent geometry and forward-checks each accepted event.
The scalar corrected planner remains the reference implementation.

`summary.csv` keeps requested denominators, accepted counts, conditional time
statistics, cell-area-weighted means, cost and 72 hour diagnostics. For the
original version, `strict_accepted_count` is the **original-clock** geographic
criterion. Do not confuse it with `continuous_accepted_count` in
`continuous_clock_audit.csv`, which independently reconstructs longitude under
the **common continuous clock**. Original reported means may include failed
geographic requests. Pairwise response changes are exported separately on
common geographic-pass subsets. These are nominal model checks, not executed
Cartesian maneuver validation or one-defect-at-a-time ablations.

Area weighting uses spherical cells clipped at the common latitude boundaries.
It is not an ellipsoidal surface-area or target-demand distribution. Area
weights are undefined for the invalid ±97 degree request domain. Failures
remain in requested denominators and have no finite response value.

## Data and plots

Large per-target NPZ chunks are distributed in the accompanying data archives.
Each preserves tolerance × latitude × longitude arrays, all requested
coordinates, success and reachability states, timing components, branch/wrap
labels, and nominal geographic errors for both versions when configured.
The manifest and checksums identify the numerical sources and saved bytes.

The existing `reconfig_plot_from_csv.py` still reads its four time columns.
It now accepts exact radii and a matched saved baseline; different spatial
grids are not merged. `reconfig_plotter_multicore.py` adds saved-data entry
points for tolerance, altitude and maps, preserving component layouts,
coastlines and the initial ground track. Cartopy remains optional for these
saved-grid maps: an included public-domain Natural Earth land GeoJSON supports
the same Plate carrée layout without downloads. The legacy live-solver map
still requires Cartopy. No figure generation triggers a new numerical run.

The full dense results supersede the sixty-target pilot as primary manuscript
evidence. More accurate outputs need not have uniformly shorter times:
rejecting a near miss may delay accepted access, while recovering a missed
band entry can make it earlier. Representative grid differences quantify
sampling sensitivity; near-zero masks, additional request states, continuous
altitude optimization and executed Cartesian propagation remain separate work.
