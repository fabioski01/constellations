"""Atomically finalize gzip containers from checksummed canonical case files.

No maneuver is planned or propagated. Every output is decoded and compared
with the original checkpoint bytes before and after atomic replacement.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256(content):return hashlib.sha256(content).hexdigest()


def finalize(job):
    folder,identifier,targets,checkpoints=job
    contents=[]
    for target in targets:
        entry=checkpoints[(identifier,target['case_index'])]
        content=(folder/entry['path']).read_bytes()
        if sha256(content)!=entry['sha256']:raise ValueError('Canonical checkpoint mismatch.')
        contents.append(content)
    expected=b''.join(contents)
    path=folder/'experiments'/identifier/'case_evidence.jsonl.gz'
    temporary=path.with_name(path.name+'.atomic.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='case_evidence.jsonl',fileobj=raw,mode='wb',mtime=0) as zipped:
            for content in contents:zipped.write(content)
        raw.flush();os.fsync(raw.fileno())
    with gzip.open(temporary,'rb') as handle:
        if handle.read()!=expected:raise ValueError('Temporary gzip round-trip mismatch.')
    os.replace(temporary,path)
    with gzip.open(path,'rb') as handle:
        if handle.read()!=expected:raise ValueError('Final gzip round-trip mismatch.')
    return dict(experiment_id=identifier,records=len(contents),jsonl_sha256=sha256(expected),
                gzip_sha256=sha256(path.read_bytes()),checkpoint_bytes_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,action='append',required=True)
    args=parser.parse_args();jobs=[];by_folder={}
    for folder in args.data_dir:
        folder=folder.resolve()
        if not json.loads((folder/'completion.json').read_text())['complete']:
            raise ValueError('Only completed campaigns may be finalized.')
        manifest=json.loads((folder/'manifest.json').read_text())
        entries=[json.loads(line) for line in (folder/'checkpoint.jsonl').read_text().splitlines()]
        checkpoints={(r['experiment_id'],r['case_index']):r for r in entries}
        expected={(e['experiment_id'],t['case_index']) for e in manifest['experiments'] for t in manifest['targets']}
        if len(entries)!=len(checkpoints) or set(checkpoints)!=expected:
            raise ValueError('Checkpoint index is incomplete or duplicated; finish/resume the numerical job before finalization.')
        for entry in manifest['experiments']:
            jobs.append((folder,entry['experiment_id'],manifest['targets'],checkpoints))
        by_folder[folder]=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job,result in zip(jobs,pool.map(finalize,jobs)):
            by_folder[job[0]].append(result)
    for folder,results in by_folder.items():
        report=dict(finalized=True,scientific_record_bytes_unchanged=True,experiments=results,
                    finalizer_source_sha256=sha256(Path(__file__).read_bytes()))
        (folder/'archive_finalization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(finalized=True,settings=len(jobs),all_checkpoint_and_gzip_bytes_verified=True),indent=2))


if __name__=='__main__':main()
