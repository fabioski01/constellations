"""Re-run source equivalence checks after workspace recovery."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper1_dense_checks import CASES, EPOCH, clock_check, scalar_check
from paper1_dense_solver import original_latitude


def historical_check():
    path = Path(__file__).resolve().parent / (
        "results/reconfig/times_deltaA/"
        "TimesAvgDeltaA_2x3_GOM6871_ROM6874_inc50.0_band49.0_dA-200to200_s10.csv")
    archived = pd.read_csv(path)
    errors = []
    for row in archived.itertuples():
        outputs = [original_latitude(CASES[0], row.delta_a_km, EPOCH, latitude,
                                     [-180., -60., 60.], [.25]) for latitude in [-49., 49.]]
        means = [float(np.mean(np.concatenate([d[key][0] for d in outputs])))
                 for key in ["response_s", "wait_s", "drift_s"]]
        errors.append(np.asarray(means) - [row.avg_total_s, row.avg_wait_s, row.avg_drift_s])
    maximum = np.max(np.abs(errors), axis=0)
    if np.any(maximum > .01):
        raise AssertionError("Historical CSV reconstruction differs.")
    return dict(offsets=len(archived), requested_targets_per_offset=6,
                maximum_component_difference_s=maximum.tolist(),
                source_file=str(path.relative_to(Path(__file__).resolve().parent)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clock = clock_check()
    tasks = []
    for case in CASES:
        for offset, latitude in [(50., 27.2), (0., -38.1),
                                 (case["a_r_km"] - case["a_g_km"], 5.4)]:
            tasks.append((case, offset, latitude, [-120., 37.], [.001, .25]))
    tasks.extend([(CASES[1], -200., -49., [-180., 0.], [.25, 5.]),
                  (CASES[1], 200., 49., [-60., 120.], [.001, .25]),
                  (CASES[0], 5., 16.3, [-120., 60.], [.25]),
                  (CASES[0], -5., -16.3, [-120., 60.], [.25]),
                  (CASES[4], 50., 85., [-60., 120.], [.25]),
                  (CASES[4], 50., 97., [-60., 120.], [.25]),
                  (CASES[0], 180., -49., [-60.], [.25])])
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scalar_check, task) for task in tasks]
        for future in as_completed(futures):
            new_rows, seconds = future.result()
            rows.extend(new_rows)
            pd.DataFrame(rows).to_csv(args.output_dir / "scalar_checks.csv", index=False)
            print(json.dumps(dict(completed_comparisons=len(rows), group_seconds=seconds)), flush=True)
    mismatches = [r for r in rows if r["scalar_success"] != r["batch_success"] or
                  (r["response_difference_s"] is not None and abs(r["response_difference_s"]) > .01)]
    report = dict(scalar_comparisons=len(rows), mismatches=mismatches,
                  maximum_response_difference_s=max(abs(r["response_difference_s"] or 0.) for r in rows),
                  legacy_clock=clock, historical_csv=historical_check(),
                  scope="Nominal source equivalence; not executed Cartesian propagation.")
    (args.output_dir / "source_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if mismatches:
        raise SystemExit("Scalar comparison failed.")


if __name__ == "__main__":
    main()
