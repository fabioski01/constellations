# Paper 1: corrections, verification, and remaining analysis

This branch starts from repository commit `4e06cbe8df79d0d525df79e5a405ee67ead633d8`. It repairs nominal Earth wait-and-drift targeting and adds reproducible exports. The committed 2026-10-14 study is a small, newly declared experiment. It does not reproduce the original thesis ensembles or validate an executed spacecraft trajectory.

## Corrections implemented

- Both timing planners accept a candidate only when its independently reevaluated nominal longitude and latitude errors, nonnegative coast times, coast caps, and optional total response horizon satisfy the request. A near miss is retained as a failure diagnostic. It cannot beat a later valid branch or enter an accepted mean.
- Latitude inputs outside ±90° are invalid. Reachability uses the actual inclination, including retrograde geometry. At 98° inclination the retained geodetic conversion gives a maximum latitude of approximately 82.052869°. The former ±97° grid is rejected. Equatorial targeting is explicitly unsupported by this isolated-crossing formulation.
- Fractional seconds and timezone offsets are preserved. Root searches use a continuous Earth angle anchored to GMST at the request epoch. Final geographic residuals are wrapped; the root function has no artificial daily angle discontinuity. UT1 is still approximated by UTC.
- Equal-radius transfers have zero duration. When drift and regional radii coincide, access can occur after regional insertion. When global and drift radii coincide, the initial coasts form one physical coast before departure. Separate wait/drift values remain bookkeeping quantities constrained by their declared caps.
- Zero-wait and zero-drift endpoints are included. Band-entry roots are placed slightly inside the requested band to prevent floating-point rounding from discarding an earlier valid candidate; final acceptance uses the original tolerance.
- Returned branch records no longer contain a circular self-reference. Importing the multicore module no longer parses another program's command-line arguments.

The existing secular mean-motion convention and 15/1 repeat condition were preserved. The gravitational parameter, J2, and 6371 km reference radius were not retuned. Nominal latitude conversion is the retained flattening-based model; its tiny algebraic residual is not an accuracy claim for an osculating Cartesian trajectory.

## Run and inspect

From the repository root, in Python 3.10 or later:

```bash
python -m pip install -r requirements-paper1.txt
python -m unittest discover -s tests -v
python paper1_analysis.py --config configs/paper1_smoke.json --output-dir results/paper1/my-run --workers 4
```

Use an empty output directory. Outer workers distribute target requests; each request uses one inner solver worker. The new runner needs NumPy, pandas, and Matplotlib. It does not require Cartopy or SGP4. Full legacy map/visualizer execution still needs its original dependencies and has not been validated here.

Both planner constructors now expose `wait_time_cap_hours`, `response_horizon_hours`, `scan_discovery`, `scan_bracket`, and `longitude_neighbor_pad`. Existing positional arguments remain compatible. An uncapped wait (`None`) requires a finite response horizon. For a scientific comparison, pass one explicit tolerance, for example `solve_min_time([0.25])`. Serial and multicore legacy defaults retain different wait caps, scan settings, and tolerance ladders; do not compare their defaults as matched experiments. The legacy `max_wrap` argument remains for compatibility; the phase range is derived from time caps.

The runner's JSON config declares the request epoch, initial node and phase, repeat ratio, targets, initial orbit cases, sampled drift offsets, tolerance, coast caps, response horizon, scan settings, and outbound budgets. Change `tolerance_offset_km` to study other offsets. Include `"regional"` in `offsets_km` to evaluate the exact regional coincidence. The five cases are E1: 500 km/50°, E2: 700 km/50°, E3: 1000 km/50°, E4: 700 km/60°, and E5: 700 km/98°.

## New saved evidence

Results are in [verification-20261014](../results/paper1/verification-20261014/). The run used nine equally weighted common targets (latitudes −40°, 0°, 40°; longitudes −120°, 0°, 120°), six offsets per case, a 72 h response horizon, 200 h wait cap, and 300 h drift cap. Most altitude requests use a 0.25° longitude tolerance. These initial states and settings are declared here, not inferred to be the original historical settings.

| Component | Requests | Accepted | Infeasible within the declared search | Numerical failures |
|---|---:|---:|---:|---:|
| Five-case altitude study, joint waiting/drift | 270 | 213 | 57 | 0 |
| E2 altitude study, immediate departure | 54 | 3 | 51 | 0 |
| E2 +50 km tolerance study, six bands | 54 | 50 | 4 | 0 |
| Total | 378 | 266 | 112 | 0 |

Nineteen regression tests passed. These cover the previously invalid Los Angeles branch selection, retrograde reachability, fractional time, omitted legs, zero-duration endpoints, regional coasting, deadlines, serial/process agreement, tolerance nesting, linear wrap bounds, and failure-preserving exports. The previously reported earlier Los Angeles near miss is now rejected; the valid later ascending solution is retained.

The separate exact-point reference implements the manuscript's two-row constant-rotation timing system and horizon bounds. All 54 E2 fixed-altitude cases agreed with enumeration expanded by three integer wraps in each row. The maximum accepted equation residual was approximately 2.84 × 10⁻¹⁴ rad. Coincident global/drift orbits use a one-coast reduction instead of a pseudoinverse. This check concerns the linear exact-point model, not the calendar-GMST longitude-band search or a continuous-altitude optimum.

The longitude-band study had no per-target tolerance-nesting violations. The independent CSV audit verified acceptance, timing sums, caps, deadlines, summary counts/means, policy nesting for mutually accepted requests, and agreement between the saved manifest hashes and the source files. Maximum nominal latitude residual was approximately 2.45 × 10⁻¹² degrees. The largest longitude error was just inside 5° in the deliberately wide-band study; each row was checked against its own requested tolerance.

| File | Contents |
|---|---|
| `manifest.json` | Complete declared config, exact regional roots, constants, package versions, and source SHA256 hashes |
| `per_target.csv` | Every request, status, errors, wait/drift/transfer times, physical departure and insertion times, wraps, cost, and diagnostics |
| `summary.csv` | Requested/accepted counts, feasible fractions, numerical-failure counts, conditional means, median, p90, maximum, and achieved longitude errors |
| `departure_paired.csv` | Same-target budget comparisons and time gains only when both policies are feasible |
| `linear_reference_checks.csv` | Bounded versus enlarged exact-point enumeration, conditioning, and equation residuals |
| `checks.json`, `audit.json` | Automated run checks and independent saved-output audit |
| Three `*_smoke.png` plots | Altitude, tolerance, and departure-policy previews, each showing conditional response and accepted fraction |
| `historical_csv_audit.json` | Limits of the original aggregate CSV |

Joint and immediate policies search the same six sampled offsets. Immediate departure removes all physical initial coasting; at global/drift coincidence this also forces drift time to zero. The direct comparator is immediate G→R transfer followed by regional coasting, obtained from the exact regional-offset request. Every policy uses the same targets, geographic acceptance, horizon, and outbound cost definition. No return maneuver, stationkeeping, or cost margin is included.

For E2, joint-policy accepted counts were 9/9 at each 110, 220, and 400 m/s budget; immediate-departure counts were 0/9, 2/9, and 3/9. Direct counts were 0/9 throughout. The direct baseline therefore provides no paired time gain in this tiny ensemble. Plot means use each policy's accepted subset and must not be subtracted as a paired improvement; use the explicit paired columns. These counts illustrate why feasibility must accompany time, not a general performance conclusion.

## Existing material to recover

The repository contains one waiting/drift CSV: `results/reconfig/times_deltaA/TimesAvgDeltaA_2x3_GOM6871_ROM6874_inc50.0_band49.0_dA-200to200_s10.csv`, source Git blob `5c0230b14dd7aab985f9882d96822710bc351a60`. It has 41 offsets and four aggregate columns, with a filename indicating six targets. Its reported peak is approximately 344.55 h at zero offset. It contains no per-target errors, accepted counts, or complete run configuration, so strict historical acceptance cannot be reconstructed from it. It is much smaller than the thesis ensembles. Existing plots preserve visual evidence but do not fill these missing records.

Recover the original per-target files, exact request epochs/node/phases, grids and weights, caps, regional radii, tolerances, source versions, and logs. Historical averages and images must remain labeled historical until either their individual candidates can be reprocessed or the studies are rerun. The old plotting wrappers still export four-column averages; use the new runner for auditable numerical studies.

## Paper destinations and work still needed

| Paper destination | Completed on this branch | Recover existing material | New analysis still needed |
|---|---|---|---|
| Methods / Figure 1 / A1 | Strict acceptance, coordinate rejection, elapsed time, horizon option, and omitted-leg event definitions | Historical settings and exact radii | Rerun all retained historical ensembles with these corrections |
| Solver verification / A2 | Linear horizon bounds, enlarged-enumeration checks, boundary and endpoint regressions | Original run/search settings | Enlarge tested states/horizons; vary scan density, neighbor padding, and altitude-grid resolution; compare the band solver against an independent method |
| Figures 2 and 4 | Small common-target altitude/inclination prototype | Original altitude/inclination outputs | Converged common-target comparisons plus separately labeled full-domain ensembles |
| Figure 3 | Reachable grid and failure export support | Original map arrays, inputs, and figure settings | Matched +50/−200 km maps with identical grids, masks, counts, scales, and reference tracks |
| Figure 5 / A5 | Small fixed-target tolerance sweep with achieved errors and nesting checks | Original tolerance records | Multiple offsets, converged ensembles, and search-resolution checks |
| Figure 7 / A3 | Three-policy definitions, outbound-budget filtering, per-target feasibility, and paired export | Reference-case provenance | Dense common altitude grids, more targets/epochs, paired gain distributions, and feasibility curves |
| Figure 6 / A4 | No executed-maneuver validation claimed | Original illustration's state, generator, frames, forces, and exact regional radius | Continuous Cartesian propagation through actual impulses, consistent Earth-fixed conversion, access/time/terminal-state errors, and repeat closure |
| Mechanism discussion / A6 | Full secular phase rate retained | Relevant original force settings | Controlled J2-on/J2-off comparison with clearly declared fixed or redesigned regional orbits |
| Study protocol / A7 | A complete manifest for the new run | Complete historical manifests | Final archive and publication figures after the larger studies |

For search-convergence work, copy the config, vary `scan_discovery` and `scan_bracket` (for example 80, 160, 320), then enlarge `longitude_neighbor_pad`; retain each run in a separate directory and compare the same per-target earliest accepted times and statuses. For altitude resolution, refine around the best sampled offsets and include exact orbit coincidences. A stable aggregate mean alone does not establish per-target convergence.

The existing visualizer still uses independently seeded SGP4 segments. Repairing that path and resolving its Earth-angle/frame convention is a separate continuous-maneuver task. Nominal solver agreement must not be presented as propagation accuracy, operational feasibility, or a globally optimal maneuver across continuous altitudes. Replace provisional manuscript numbers only after the corresponding converged, declared study has been completed.
