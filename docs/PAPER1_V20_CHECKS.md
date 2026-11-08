# Manuscript v20 numerical checks

The numerical core is taken from commit
`0443aa0e98666cb6531ddb0c97d2840b32a6e47f` on `paper1/solver-corrections`.
The new scripts do not alter the planner, force model, command law, or
acceptance criterion.

Run from the repository root with the existing Paper 1 dependencies:

```bash
python -m unittest discover -s tests -v
python paper1_check_v20.py
python paper1_v20_figures.py
python paper1_altitude_figure_report.py --extend-negative-offsets --output-dir results/paper1/v20-altitude --source-commit 0443aa0e98666cb6531ddb0c97d2840b32a6e47f --workers 4
python paper1_dense_audit.py --data-dir results/paper1/v20-altitude --output-dir results/paper1/v20-altitude/audit
python paper1_validate_maneuver.py --config configs/paper1_validation_los_angeles.json --output-dir results/paper1/v20-los-angeles-replay
```

The completed check covers all 94 altitude configurations and 6,091,200
coordinate requests, all seven six-band tolerance sweeps, the 324-target
nominal direct/excursion comparison, the saved 648-target propagation CSV,
and the Los Angeles connected propagation example. All 54 existing tests
pass. The independent altitude reconstruction has no failures. Pilot CSV
SHA256 is `000ffacc546f74a661092b0d2f6d70b64757c57aaaccd415cbba26150a2ffaa2`.
The Los Angeles replay reproduces the saved numerical evidence exactly.
The complete 21-setting Cartesian sensitivity campaign is not rerun.

The narrow E2 tolerance change from 0.001 to 0.005 degree reduces mean
response by 153.11 s. Increasing to 0.25 degree reduces the mean by
3.07719 h but the median by only 60.5793 s; 5,743 targets switch branch
or angular wraps. These values reproduce the manuscript.

In the 324-target +100 km comparison, 32 selected plans have zero drift
duration. Combining the opposing coincident impulses gives 159.259158 m/s
for those plans and 211.700332 m/s for the others, with mean 206.520957 m/s.
The direct transfer remains 106.645 m/s.

The nominal observation comparison uses the same full-precision regional
root, targets, initial state, geographic band, and coast caps for both
Earth rotation conventions. Both are replanned with the primary batched
finite-band solver. At 72 h, uniform rotation gives mean visit counts of
2.616698 direct and 2.135154 excursion; calendar GMST gives 1.137068 and
1.147515. The unchanged initial orbit gives 0.121636 in either calculation
at this precision. Nine coordinates, three strategies, four windows, and
two rotation conventions give 216 independent scalar root-location
checks, all agreeing with the event enumeration.

The large count change comes from strict classification near a band edge:
small Earth-angle differences exceed the tiny planning margin. It does
not imply a large physical ground-track displacement. The comparison
does not demonstrate propagated repeated access or scientific retrieval
performance. A physical planning margin or correction must be evaluated
against the original geographic criterion before claiming achieved visits.

Detailed arrays and figure data are saved locally by the scripts. They
are not added to the repository. The manifest records exact source hashes
and the scope of the recomputation.
