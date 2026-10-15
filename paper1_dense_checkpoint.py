"""Verify or recover checkpoint indexing from completed atomic NPZ shards.

Some execution environments replace an open file's path while a long-lived
append descriptor is still writing. Saved atomic shards remain independently
usable. Recovery requires an otherwise complete run, unchanged source/config,
exact saved coordinates/shapes, and matching checksums for every existing
index entry. Newly indexed shards are explicitly marked; unknown runtimes
are not invented. Numerical arrays are never modified by this utility.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--recover-index',action='store_true')
    args=p.parse_args();folder=args.data_dir
    m=json.loads((folder/'manifest.json').read_text());c=json.loads((folder/'completion.json').read_text())
    if not c['complete']:raise ValueError('Recovery is permitted only after the declared run is complete.')
    root=Path(__file__).resolve().parent
    for name,sha in m['source_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('Source changed: '+name)
    index=folder/'checkpoint.jsonl'
    existing=[json.loads(line) for line in index.read_text().splitlines()]
    known={(x['experiment_id'],x['latitude_start']):x for x in existing}
    rows=[];recovered=0;expected_files=set()
    for e in m['experiments']:
        latitude=np.linspace(-e['latitude_band_deg'],e['latitude_band_deg'],e['n_lat'])
        longitude=m['config']['longitude_start_deg']+np.arange(e['n_lon'])*360/e['n_lon']+e['longitude_shift_deg']
        for first in range(0,e['n_lat'],m['config']['latitude_chunk_size']):
            last=min(first+m['config']['latitude_chunk_size'],e['n_lat'])
            filename=f'{e["experiment_id"]}_lat{first:03d}.npz';path=folder/'per_target'/filename
            expected_files.add(filename)
            with np.load(path) as arrays:
                for key,value in [('latitude_deg',latitude[first:last]),('longitude_deg',longitude),
                                  ('tolerances_deg',np.asarray(e['tolerances_deg']))]:
                    if not np.array_equal(arrays[key],value):raise ValueError('Coordinate mismatch: '+filename+' '+key)
                for version in ['corrected','original'] if e['original'] else ['corrected']:
                    shape=(len(e['tolerances_deg']),last-first,e['n_lon'])
                    for field in ['response_s','wait_s','drift_s','success','longitude_error_deg','latitude_error_deg','branch','phase_wrap','longitude_wrap']:
                        if arrays[version+'_'+field].shape!=shape:raise ValueError('Array shape mismatch: '+filename)
                accepted=int(arrays['corrected_success'].sum())
            sha=digest(path);old=known.get((e['experiment_id'],first))
            if old:
                if old['sha256']!=sha:raise ValueError('Indexed shard checksum changed: '+filename)
                row=old
            else:
                recovered+=1
                row=dict(experiment_id=e['experiment_id'],latitude_start=first,latitude_stop=last,
                    relative_path=filename,sha256=sha,runtime_s=None,
                    configured_targets=(last-first)*e['n_lon']*len(e['tolerances_deg']),
                    corrected_accepted=accepted,index_recovered_from_atomic_shard=True)
            rows.append(row)
    actual_files={p.name for p in (folder/'per_target').glob('*.npz')}
    if actual_files!=expected_files:raise ValueError('Unexpected or missing per-target shards.')
    report=dict(checked_chunks=len(rows),previously_indexed_chunks=len(known),recovered_chunks=recovered,
        complete_sources_coordinates_shapes_and_existing_checksums_verified=True,
        numerical_arrays_modified=False,checked_utc=datetime.now(timezone.utc).isoformat())
    if recovered:
        if not args.recover_index:raise ValueError(f'{recovered} atomic shards are unindexed; use explicit --recover-index after reviewing the recovery scope.')
        backup=folder/'checkpoint_before_recovery.jsonl'
        if backup.exists():raise ValueError('A recovery backup already exists; inspect before another recovery.')
        backup.write_bytes(index.read_bytes())
        temporary=folder/'checkpoint_recovered.tmp'
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows));temporary.replace(index)
        (folder/'checkpoint_recovery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
