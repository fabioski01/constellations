# Controlled propagation sensitivity around a +100 km reference

The three sweeps vary longitude tolerance, drift semimajor-axis offset, and
the five orbit configurations already used in the nominal paper. They retain
the connected Cartesian gravity-plus-J2 validation worker, numerical-mean
initialization convention, fixed signed RTN transverse Hohmann impulses,
request epoch, initial mean node and phase, coast caps, and acceptance rules.
No targeting correction, empirical margin, event retiming, or extra force is
introduced. This is a descriptive physical-model comparison, not operational
mission validation or causal isolation of the discrepancy sources.

The reference is 700 km initial mean altitude, 50 degrees inclination,
**+100 km drift offset** and 0.25 degree longitude tolerance. The smaller
common grid has 18 periodic longitudes and the same 18 latitude rows as the
previous 36 by 18 run, including +/-49 degree endpoints: 324 coordinates.
Cells have the same clipped spherical weighting convention; equal-coordinate
fractions are primary. The 180 degree endpoint is not duplicated.

## Sampled settings

- Tolerance: 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 1 and 5 degrees;
  initial orbit and +100 km drift offset are fixed. Each target is re-planned
  before propagation, rather than merely rescored against a different band.
- Drift offset: -200, -100, -50, -10, 0, +10, +50, +100 and +200 km,
  plus exact drift/regional coincidence computed from the repeat root;
  initial orbit is 700 km / 50 degrees and tolerance is 0.25 degrees.
- Orbit cases: 500, 700 and 1000 km at 50 degrees, and 700 km at
  60 and 98 degrees; drift offset is +100 km and tolerance is 0.25 degrees.
  The regional 15/1 orbit is recomputed at each inclination. These are two
  one-variable slices, not a full altitude/inclination factorial study.

The shared reference is computed once, giving 21 unique settings and 6804
coordinate-setting requests on the screening grid. Nominally infeasible
targets remain in the denominator, including at global/drift coincidence.

## Reproduce

```bash
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper1_validate_sensitivity.py \
  --config configs/paper1_validation_sensitivity.json \
  --output-dir results/paper1/validation-sensitivity-plus100 --workers 8
python paper1_sensitivity_report.py \
  --data-dir results/paper1/validation-sensitivity-plus100
```

`--probe` runs six geographic controls per setting into a separate output.
`--resume` requires byte-identical numerical sources and inputs and verifies
every existing checkpoint's SHA256 before continuing. Completed records are
written atomically and individually indexed. A numerical or convergence
failure is retained and makes the final run unsuccessful.

`--reuse PATH` can consume a previous compatible single-grid or sensitivity
campaign. It requires identical physical inputs and Cartesian initial state,
unchanged hashes of every numerical worker/source, and an exactly matching
target coordinate. Case indices and cell weights are updated for the new
grid; physical evidence remains unchanged and retains its origin archive
hash and original index. This avoids recomputing the existing +50 km requests
and nested-grid requests. Reuse is never allowed across different tolerances
or different maneuvers. Full evidence archives are distributed separately.
Reuse a campaign only after it has completed. When other settings are still
running, isolate the manifest and archive of a fully completed setting first;
an archive being written is not a completed input dataset.

## Selected refinement

```bash
python paper1_validate_sensitivity.py \
  --output-dir results/paper1/validation-sensitivity-refinement-plus100 \
  --n-longitude 36 --only EXPERIMENT_ID_1 EXPERIMENT_ID_2 \
  --reuse results/paper1/validation-sensitivity-plus100 --workers 8
python paper1_sensitivity_report.py \
  --data-dir results/paper1/validation-sensitivity-plus100 \
  --refinement-dir results/paper1/validation-sensitivity-refinement-plus100
```

Pass `--refinement-dir` more than once to compare separately saved refinements.

After each numerical process has exited, finalize its compressed containers
from the SHA256-checked canonical case files and independently audit the result:

```bash
python paper1_finalize_sensitivity.py --data-dir RESULTS
python paper1_verify_sensitivity.py --data-dir RESULTS
```

The finalizer writes containers atomically and verifies both decoded bytes
and checkpoint hashes; it never plans or propagates a maneuver. A prior
completion file can remain present while a resume is active, so wait for that
process to exit before starting dependent finalization or readback.

The common +100 km reference and selected settings with prominent errors or
acceptance changes are checked on the 36 by 18 grid. These checks assess
longitude sampling at the same latitude rows; they are not a latitude-grid
convergence test. Selection and observed changes are documented with results.

## Outputs and definitions

`all_maneuvers.csv` retains every requested coordinate-setting pair.
`experiments/ID/per_maneuver.csv` contains each setting independently;
`sensitivity_summary.csv` includes sample counts, both requested-coordinate
and nominally-accepted denominators, spherical cell-area fractions, error
distributions, timing, terminal-orbit and repeat diagnostics.
`comparison_table.csv` is the compact comparison used for paper drafting.

The primary figures show longitude prediction MAE and 95th absolute
percentile, longitude-only and full matched-event acceptance, extended-window
access, nominal feasibility, and signed event timing. Coincident-orbit points
are explicitly marked because a transfer disappears and insertion/access
timing changes. Sampled values are connected only as a visual guide.

Prediction error compares propagated and nominal longitude at corresponding
same-branch latitude crossings. Acceptance instead compares propagated
longitude with the target and uses the original requested tolerance.
Full matched-event acceptance additionally requires insertion eligibility.
Extended-window access can be delayed and does not establish accuracy of
the planned response. Terminal numerical-mean semimajor-axis error and
15-node repeat closure are separate diagnostics with no invented pass limits.

Every newly propagated request is executed twice with the existing primary
and tighter DOP853 settings; the 0.1 m position, 0.001 s event-time, event-count
and acceptance-classification convergence gates are unchanged.

Implementation source is saved on the research branch. The complete result
package is distributed privately: readable per-maneuver and aggregate CSV/JSON,
scientific raster/vector figures, reports and full event/burn evidence.
Public release of research datasets requires separate explicit authorization.

The optional `paper1_sensitivity_evidence.py` utility creates one private
lossless MessagePack/XZ bundle. Install `requirements-paper1-evidence.txt` to
read it. The bundle includes full origin archives for reused records and
all source/checkpoint metadata, without dropping events or burn states.
Restoration is checked against the original JSONL and gzip SHA256 hashes:

```bash
python paper1_sensitivity_evidence.py restore ARCHIVE.tar.xz --verify-only
python paper1_sensitivity_evidence.py restore ARCHIVE.tar.xz --destination REPOSITORY_ROOT
```

The JSONL bytes preserve every scientific value independently of transport
compression. Reproducing gzip bytes also requires a compatible Python/zlib
runtime, recorded in the archive. Existing differing files are never overwritten.
