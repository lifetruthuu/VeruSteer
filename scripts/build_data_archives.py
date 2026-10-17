#!/usr/bin/env python3
"""Build deterministic, ordinary Git data shards from an expanded artifact."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"corpus": ["data"], "selector": ["selector/embedding_table", "selector/idmap"]}


def build(name, root, output, chunk_bytes):
    roots = GROUPS[name]
    paths = sorted(p for item in roots for p in (root / item).rglob('*')
                   if p.is_file() and p.name != 'README.md')
    if not paths:
        raise ValueError(f"Restore {name} before rebuilding its archive")
    if any(p.is_symlink() for p in paths):
        raise ValueError("Data packages must not contain symlinks")
    package = {"name": name, "roots": roots, "file_count": len(paths),
               "unpacked_bytes": sum(p.stat().st_size for p in paths), "parts": []}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pack-', dir=output) as tmp:
        archive = Path(tmp) / 'payload.tar.gz'
        with archive.open('wb') as compressed:
            pigz = shutil.which('pigz')
            proc = subprocess.Popen([pigz, '-6', '-n', '-p', '4'], stdin=subprocess.PIPE, stdout=compressed) if pigz else None
            stream = proc.stdin if proc else gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=6, mtime=0, filename='')
            try:
                with tarfile.open(fileobj=stream, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                    for path in paths:
                        info = tar.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        info.mode = 0o644
                        info.pax_headers = {}
                        with path.open('rb') as source:
                            tar.addfile(info, source)
            finally:
                stream.close()
                if proc and proc.wait() != 0:
                    raise RuntimeError('Compression failed')
        package['compressed_bytes'] = archive.stat().st_size
        with archive.open('rb') as source:
            index = 1
            while chunk := source.read(chunk_bytes):
                name_part = f'{name}-{index:03d}.bin'
                (output / name_part).write_bytes(chunk)
                package['parts'].append({'path': 'downloads/' + name_part, 'bytes': len(chunk)})
                index += 1
    print(f"{name}: {package['unpacked_bytes']:,} -> {package['compressed_bytes']:,} bytes; {len(package['parts'])} shards", flush=True)
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk-mib', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.chunk_mib <= 48:
        parser.error('--chunk-mib must be between 1 and 48')
    output = ROOT / 'downloads'
    if list(output.glob('*.bin')):
        parser.error('Remove the previous downloads/*.bin shards before rebuilding')
    packages = [build(name, ROOT, output, args.chunk_mib * 1024 * 1024) for name in GROUPS]
    manifest = {'format': 'tar.gz-split-v1', 'chunk_bytes': args.chunk_mib * 1024 * 1024,
                'packages': packages, 'verification': 'Shard sizes, gzip CRC, archive member counts and sizes'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
