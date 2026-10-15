"""Run a declared, independently complete part of the dense study.

The numerical solvers and orbit settings are unchanged. Corrected grids are
saved before the slower matched original-code comparison. Atomic per-chunk
receipts avoid relying on a long-lived checkpoint file descriptor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from paper1_dense_analysis import (SOURCES, atomic_json, chunk_worker, digest,
                                   experiments, summarize_experiment)

ROOT = Path(__file__).resolve().parent


def selected_experiments(config, stage):
    all_items = experiments(config)
    if stage == "corrected":
        profiles = {"common", "coarse", "medium", "shifted", "native_requested"}
        selected = [dict(e, original=False) for e in all_items if e["profile"] in profiles]
    else:
        selected = [e for e in all_items if e["profile"] == "common" and
                    (e["offset_km"] == 50. or
                     (e["case"]["case_id"] == "E2" and e["offset_km"] in [-200., 200.]))]
    # Finish the main tolerance comparison in all five cases first.
    return sorted(selected, key=lambda e: (
        0 if e["profile"] == "common" and e["offset_km"] == 50. else
        1 if e["profile"] == "common" else 2,
        e["case"]["case_id"], e["experiment_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper1_dense_360x180.json"))
    parser.add_argument("--stage", choices=["corrected", "paired"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    scope = (
        "Corrected grids: five cases, fourteen drift choices on the common 360x180 domain; "
        "six longitude tolerances at +50 km and E2 +/-200 km; representative resolution "
        "and longitude-shift checks; native requested-domain diagnostics. No original-code "
        "results, historical figure reconstruction or refined-altitude sweep in this stage."
        if args.stage == "corrected" else
        "Matched original/corrected grids at +50 km in all five cases and +/-200 km in E2, "
        "each at six longitude tolerances on identical 360x180 common-domain coordinates. "
        "This stage does not reproduce every historical figure or compare all drift choices."
    )
    config = dict(config, scope=scope, recovery_stage=args.stage)
    items = selected_experiments(config, args.stage)
    sources = {name: digest(ROOT / name) for name in SOURCES + [Path(__file__).name]}
    identity = dict(config=config, source_sha256=sources, experiments=items)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shards = output / "per_target"
    receipts = output / "chunk_receipts"
    shards.mkdir(exist_ok=True)
    receipts.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output exists: use --resume or another directory.")
        old = json.loads(manifest_path.read_text())
        if any(old[key] != value for key, value in identity.items()):
            raise ValueError("Configuration, scope or numerical source changed.")
    else:
        atomic_json(manifest_path, dict(identity,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            grid_order="array axes tolerance, latitude, longitude", scope=scope))
    completed = {}
    for receipt in receipts.glob("*.json"):
        row = json.loads(receipt.read_text())
        path = shards / row["relative_path"]
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError("Saved chunk checksum failed: " + str(path))
        completed[(row["experiment_id"], row["latitude_start"])] = row
    tasks = []
    total_chunks = 0
    for item in items:
        for first in range(0, item["n_lat"], config["latitude_chunk_size"]):
            total_chunks += 1
            if (item["experiment_id"], first) not in completed:
                last = min(first + config["latitude_chunk_size"], item["n_lat"])
                path = shards / f'{item["experiment_id"]}_lat{first:03d}.npz'
                tasks.append((item, first, last, config, str(path)))

    def checkpoint():
        ordered = [completed[k] for k in sorted(completed)]
        temporary = output / "checkpoint.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in ordered))
        temporary.replace(output / "checkpoint.jsonl")
        finished = [e["experiment_id"] for e in items if all(
            (e["experiment_id"], first) in completed
            for first in range(0, e["n_lat"], config["latitude_chunk_size"]))]
        atomic_json(output / "completion.json", dict(
            finished_experiments=finished, total_experiments=len(items),
            complete=len(finished) == len(items), finished_chunks=len(completed),
            total_chunks=total_chunks, scope=scope,
            total_solver_target_pairs=sum(e["n_lat"] * e["n_lon"] *
                len(e["tolerances_deg"]) * (2 if e["original"] else 1) for e in items)))

    checkpoint()
    print(json.dumps(dict(stage=args.stage, experiments=len(items), chunks=total_chunks,
                          pending_chunks=len(tasks))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(chunk_worker, task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            atomic_json(receipts / (row["relative_path"] + ".json"), row)
            completed[(row["experiment_id"], row["latitude_start"])] = row
            checkpoint()
            if count % 18 == 0 or count == len(tasks):
                print(json.dumps(dict(finished_chunks=len(completed), total_chunks=total_chunks,
                    experiment_id=row["experiment_id"], chunk_seconds=row["runtime_s"])), flush=True)
    summary = []
    for item in items:
        rows = [completed[(item["experiment_id"], first)]
                for first in range(0, item["n_lat"], config["latitude_chunk_size"])]
        summary.extend(summarize_experiment(item, [shards / r["relative_path"] for r in rows], config))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    checkpoint()
    print(json.dumps(dict(stage=args.stage, complete=True, experiments=len(items),
        raw_bytes=sum(p.stat().st_size for p in shards.glob("*.npz")))), flush=True)


if __name__ == "__main__":
    main()
