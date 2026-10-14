# Paper 1 first grid iteration

This study preserves the initial wait, two Hohmann transfers and intermediate circular drift. Its primary questions are drift altitude versus time/cost, target location versus coast durations, and longitude tolerance versus response. The earlier nine-target verification directory remains separate.

Ten geodetic latitudes from −49 to +49 degrees and six longitudes from −180 to +120 degrees form the common grid. All five orbit cases use these same sixty points. Initial node and phase are zero at May 5, 2025, 12:00 UTC. Separate wait/drift caps are 200/300 h, matching the supplied multicore driver's caps. The 72 h statistic is exported separately; it does not censor primary results. This is not a reconstruction of every historical thesis figure.

Fourteen altitude choices per case include thirteen common offsets, extra samples around zero, and exact regional coincidence. All five cases have six-band sweeps at +50 km; E2 also has −200 and +200 km sweeps. There are 6,720 configured records and 6,300 distinct solver calculations; 420 records share an identical physical request across study labels. Of the distinct calculations, 5,974 are accepted and 326 have no accepted candidate under the declared search. No request has a numerical-failure status.

| Case | Offset km | Accepted | Coordinate mean h | Area weighted mean h |
|:--|--:|:--|--:|--:|
| E1 | +50 | 60/60 | 65.636 | 66.775 |
| E1 | +100 | 60/60 | 36.822 | 37.187 |
| E1 | +200 | 60/60 | 22.669 | 22.803 |
| E2 | +50 | 60/60 | 44.407 | 44.513 |
| E2 | +100 | 60/60 | 35.131 | 34.961 |
| E2 | +200 | 60/60 | 24.180 | 24.152 |
| E3 | +50 | 60/60 | 53.220 | 52.655 |
| E3 | +100 | 60/60 | 32.535 | 32.253 |
| E3 | +200 | 60/60 | 24.487 | 23.915 |
| E4 | +50 | 60/60 | 45.381 | 46.088 |
| E4 | +100 | 60/60 | 33.449 | 33.853 |
| E4 | +200 | 60/60 | 24.099 | 24.269 |
| E5 | +50 | 60/60 | 45.900 | 47.011 |
| E5 | +100 | 60/60 | 32.274 | 32.309 |
| E5 | +200 | 60/60 | 24.092 | 24.825 |

The E1 +100/+200 km means are 36.822/22.669 h, close to approximate historical 35/22 h readings. This is not a controlled old/new solver comparison; original full-grid records and settings are still needed for that attribution.

For E2, −200 km is earlier on 46 of the sixty paired targets and +50 km on fourteen. The mean paired reduction from selecting −200 km is 21.346 h. At +50 km the means at 0.001/0.25/5 degrees are 49.405/44.407/17.818 h, each over all sixty targets. At +200 km, widening 0.001 to 0.25 degrees saves about 59.26 s on average; at +50 km it saves about 5.00 h. Geographic alternative changes explain part of this offset-dependent sensitivity.

Primary means give every coordinate point equal weight. Secondary weights represent spherical latitude cells between midpoint boundaries clipped to ±49 degrees, with 60-degree longitude cells. Conditional area-weighted means are normalized over accepted cells. They are quadrature statistics, not measured continuous-area coverage.

Phase-wrap pruning is explicit, defaults to false in the general planner, and is enabled by this grid config. A positive-rate response lower bound permits stopping later wraps only after a candidate meets the strictest requested band. Twenty-four declared requests agree exactly with full phase enumeration. Doubling scans from 160 to 320 and enlarging padding from eight to twelve preserves acceptance and changes response by at most 2.91e−11 s on that sample. Nineteen regression tests pass.

The independent audit reconstructs nominal geography from saved timing records, checks transfers/costs and means, and confirms source hashes. It does not validate executed Cartesian dynamics.

## Reproduce and inspect

Run from the repository root with requirements-paper1.txt. Use a fresh output directory.

    python paper1_grid_analysis.py --config configs/paper1_grid_10x6.json --output-dir results/paper1/new-grid --workers 6 --checkpoint results/paper1/new-grid-checkpoint.jsonl
    python paper1_grid_checks.py --config configs/paper1_grid_10x6.json --output-dir results/paper1/new-grid --workers 2
    python paper1_grid_audit.py --data-dir results/paper1/new-grid
    python paper1_grid_figures.py --data-dir results/paper1/new-grid

An interrupted grid can resume with the same command plus --resume. It verifies configuration, repeat roots, numerical-source hashes and every completed task. The archived initial_runner.py preserves the exact first runner used before recovery; its hash matches the manifest. Recovery added checkpoint handling while numerical sources remained unchanged.

The archive contains per_target.csv, targets.csv, summary.csv, location_paired.csv, tolerance_paired.csv, interaction_summary.csv, manifests and audits. Five figures are supplied as PNG and vector PDF. Geographic cells show actual sampled coordinates without interpolation.

## Completed work and next analyses

Completed: sixty common points, five cases, fourteen altitude choices, seven tolerance sweeps, paired geographic effects, full-enumeration/refinement sample and saved-record reconstruction.

New: refine and shift the spatial grid, especially longitudes near repeat-track structure; refine altitude/tolerance near changes; test more request states; validate continuous impulses and repeat closure. Nested latitude/longitude refinement can preserve existing samples. Separately label full-reach ensembles for higher inclinations.

Recover: original records and exact configurations only where a faithful historical comparison is retained. This first 10 by 6 iteration does not establish spatial convergence, continuous-altitude optimality or spacecraft execution accuracy.
