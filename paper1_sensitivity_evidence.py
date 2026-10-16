"""Losslessly bundle or verify the full private event/burn evidence.

MessagePack sequences preserve JSON object order, binary64 floats, integers,
booleans, strings and nulls. Restoration must reproduce every original JSONL
byte and, under the recorded Python/zlib runtime, every original gzip byte.
The repository retains this reader; the private archive contains data only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import zlib

import msgpack

ROOT=Path(__file__).resolve().parent


def digest(data):return hashlib.sha256(data).hexdigest()


def safe_path(destination,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe archive target.')
    result=destination.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Archive target escapes the selected destination.')
    return result


def add_bytes(archive,name,content):
    info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644;info.mtime=0
    archive.addfile(info,io.BytesIO(content))


def identify_archive(path):
    for folder in [path.parent,path.parent.parent.parent]:
        if (folder/'manifest.json').exists():
            manifest=json.loads((folder/'manifest.json').read_text())
            if 'context' in manifest:
                scenario=manifest['context']['scenario']
            else:
                entry=next(e for e in manifest['experiments'] if e['experiment_id']==path.parent.name)
                scenario=entry['context']['scenario']
            key=tuple(scenario[k] for k in ['global_altitude_km','inclination_deg','drift_offset_km','longitude_tolerance_deg'])
            return key,folder
    raise ValueError('No source manifest for '+str(path))


def bundle(folders,output,source_commit,reader_commit=None):
    pending=[];metadata=set();products=set()
    for folder in folders:
        completion=json.loads((folder/'completion.json').read_text())
        if not completion['complete']:raise ValueError('Only complete campaigns may be bundled.')
        pending.extend(sorted(folder.glob('experiments/*/case_evidence.jsonl.gz')))
        products.update(p for p in folder.iterdir() if p.is_file() and
                        p.suffix in ['.csv','.json','.md','.png','.svg','.pdf'])
        products.update(folder.glob('experiments/*/per_maneuver.csv'))
        products.update(folder.glob('experiments/*/summary.json'))
    sources={};gzip_hashes={}
    while pending:
        path=pending.pop().resolve()
        if path in sources:continue
        key,folder=identify_archive(path)
        raw_hash=hashlib.sha256();count=0
        with gzip.open(path,'rb') as handle:
            for line in handle:
                raw_hash.update(line);count+=1;record=json.loads(line)
                origin=record.get('reused_from')
                if origin:
                    parent=ROOT/origin['archive']
                    if parent not in gzip_hashes:gzip_hashes[parent]=digest(parent.read_bytes())
                    if gzip_hashes[parent]!=origin['sha256']:
                        raise ValueError('Origin archive checksum mismatch.')
                    pending.append(parent)
        sources[path]=dict(key=key,folder=folder,raw_jsonl_sha256=raw_hash.hexdigest(),count=count,
                           gzip_sha256=digest(path.read_bytes()))
        for name in ['manifest.json','checkpoint.jsonl','completion.json','verification.json']:
            if (folder/name).exists():metadata.add(folder/name)
    # Group physically identical settings so nested-grid repetitions compress well.
    ordered=sorted(sources,key=lambda p:(sources[p]['key'],str(p)))
    index=dict(schema='paper1_lossless_event_evidence_v1',numerical_source_commit=source_commit,
               reader_source_commit=reader_commit or source_commit,
               archive_producer_sha256=digest(Path(__file__).read_bytes()),python_version=platform.python_version(),
               encoding='Consecutive MessagePack objects, one complete original JSON record each.',
               zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,msgpack_version=msgpack.__version__,archives=[],metadata=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp')
    with tarfile.open(temporary,'w:xz',preset=9) as archive:
        for number,path in enumerate(ordered):
            member=f'records/{number:04d}.msgpack'
            source=sources[path]
            with tempfile.TemporaryFile(dir=output.parent) as packed:
                packer=msgpack.Packer(use_bin_type=True)
                with gzip.open(path,'rb') as handle:
                    for line in handle:
                        record=json.loads(line)
                        encoded=packer.pack(record)
                        # The portable representation must preserve exact JSON serialization.
                        restored=msgpack.unpackb(encoded,raw=False)
                        rebuilt=(json.dumps(restored,ensure_ascii=False,allow_nan=False)+'\n').encode()
                        if rebuilt!=line:raise ValueError('MessagePack changed a JSON record.')
                        packed.write(encoded)
                size=packed.tell();packed.seek(0)
                info=tarfile.TarInfo(member);info.size=size;info.mode=0o644;info.mtime=0
                archive.addfile(info,packed)
            index['archives'].append(dict(member=member,source_relative_path=str(path.relative_to(ROOT)),
                                          record_count=source['count'],raw_jsonl_sha256=source['raw_jsonl_sha256'],
                                          gzip_sha256=source['gzip_sha256']))
            print(json.dumps(dict(packed_archives=number+1,total_archives=len(ordered))),flush=True)
        for path in sorted(metadata):
            relative=str(path.relative_to(ROOT));member='metadata/'+relative
            content=path.read_bytes();add_bytes(archive,member,content)
            index['metadata'].append(dict(member=member,target_relative_path=relative,sha256=digest(content)))
        for path in sorted(products-metadata):
            relative=str(path.relative_to(ROOT));member='products/'+relative
            content=path.read_bytes();add_bytes(archive,member,content)
            index['metadata'].append(dict(member=member,target_relative_path=relative,sha256=digest(content),file_kind='readable_result_or_figure'))
        add_bytes(archive,'bundle_index.json',(json.dumps(index,indent=2)+'\n').encode())
        readme=(
            '# Complete validation event and burn evidence\n\n'
            'This archive contains every original saved record for the screening and selected refined grids, plus complete origin archives for reused records. '
            'Portable MessagePack sequences retain all numerical values and object order; no observations, rejected cases, events or burn states are discarded. '
            'All readable CSV/JSON results, reports and raster/vector figures are included under products/. '
            'The research repository supplies the implementation and reader; these result files are distributed privately.\n\n'
            f'Numerical source: https://github.com/fabioski01/constellations/tree/{source_commit}\n\n'
            f'Reader: https://github.com/fabioski01/constellations/blob/{reader_commit or source_commit}/paper1_sensitivity_evidence.py\n\n'
            'Use the repository reader, with the msgpack Python package installed:\n\n'
            '    python paper1_sensitivity_evidence.py restore ARCHIVE.tar.xz --verify-only\n'
            '    python paper1_sensitivity_evidence.py restore ARCHIVE.tar.xz --destination REPOSITORY_ROOT\n\n'
            'Restoration checks SHA256 of the reconstructed JSONL and gzip bytes, recreates individual case checkpoints, and preserves existing identical files. '
            'It refuses to overwrite differing files. Gzip byte identity can depend on the recorded Python/zlib runtime; the uncompressed JSONL check verifies the scientific records independently.\n')
        add_bytes(archive,'README.md',readme.encode())
    os.replace(temporary,output)
    result=dict(file=str(output),bytes=output.stat().st_size,sha256=digest(output.read_bytes()),
                source_archives=len(sources),records=sum(e['record_count'] for e in index['archives']))
    print(json.dumps(result,indent=2))
    return result


def write_new_or_identical(path,content):
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Refusing to overwrite differing file: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)


def restore(path,destination,verify_only):
    count=0
    with tarfile.open(path,'r:xz') as archive:
        index=json.load(archive.extractfile('bundle_index.json'))
        for item in index['archives']:
            unpacker=msgpack.Unpacker(archive.extractfile(item['member']),raw=False,strict_map_key=True)
            raw_hash=hashlib.sha256();number=0;compressed=io.BytesIO()
            with gzip.GzipFile(filename='case_evidence.jsonl',fileobj=compressed,mode='wb',mtime=0) as zipped:
                for record in unpacker:
                    line=(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode()
                    raw_hash.update(line);zipped.write(line);number+=1
                    if not verify_only:
                        target=safe_path(destination,item['source_relative_path']).parent/'cases'/f'case_{record["row"]["case_index"]:04d}.json'
                        write_new_or_identical(target,line)
            if raw_hash.hexdigest()!=item['raw_jsonl_sha256'] or number!=item['record_count']:
                raise ValueError('Uncompressed scientific evidence checksum mismatch.')
            zipped_bytes=compressed.getvalue()
            if digest(zipped_bytes)!=item['gzip_sha256']:
                raise ValueError('JSONL records are intact, but gzip bytes differ; use the recorded Python/zlib runtime.')
            if not verify_only:write_new_or_identical(safe_path(destination,item['source_relative_path']),zipped_bytes)
            count+=number
        for item in index['metadata']:
            content=archive.extractfile(item['member']).read()
            if digest(content)!=item['sha256']:raise ValueError('Metadata checksum mismatch.')
            if not verify_only:write_new_or_identical(safe_path(destination,item['target_relative_path']),content)
    result=dict(verified=True,records=count,archives=len(index['archives']),
                all_jsonl_and_gzip_bytes_identical=True,metadata_files=len(index['metadata']))
    print(json.dumps(result,indent=2));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('bundle');make.add_argument('--data-dir',type=Path,action='append',required=True)
    make.add_argument('--output',type=Path,required=True);make.add_argument('--source-commit',required=True)
    make.add_argument('--reader-commit')
    read=sub.add_parser('restore');read.add_argument('archive',type=Path)
    read.add_argument('--destination',type=Path,default=ROOT);read.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.command=='bundle':bundle([p.resolve() for p in args.data_dir],args.output.resolve(),args.source_commit,args.reader_commit)
    else:restore(args.archive.resolve(),args.destination.resolve(),args.verify_only)


if __name__=='__main__':main()
