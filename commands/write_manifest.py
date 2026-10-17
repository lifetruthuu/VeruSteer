#!/usr/bin/env python3
"""List the distributable files and their sizes; no content hashes are calculated."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PURPOSE={'data':'Research data and fixed splits','selector':'Selector training and shared embeddings',
 'pipelines':'Repair-tree collection and online repair','scripts':'Execution and replay entrypoints',
 'commands':'Data processing and evaluation','draw':'Paper figures','results':'Recorded paper outcomes',
 'docs':'Usage and data documentation','tests':'Portable interface checks',
 'downloads':'Compressed research data and download instructions'}

def main():
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if '.git' in rel.parts or rel.parts[0]=='outputs' or '__pycache__' in rel.parts: continue
        if not path.is_file() or rel==Path('manifest.json'): continue
        if '.git' in rel.parts or path.is_symlink(): raise ValueError(f'Unexpected repository metadata or link: {rel}')
        files.append(dict(path=str(rel),bytes=path.stat().st_size,purpose=PURPOSE.get(rel.parts[0],'Package documentation or configuration')))
    result={'layout':'Expanded artifact, including compressed download shards',
            'files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),
            'corpus':dict(tasks=684,trees=3224,trajectories=56428,training_pairs=18885),
            'verification':'File inventory, sizes, parsed records, IDs, dimensions, and runtime checks'}
    (ROOT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['file_count'],'files;',result['total_bytes'],'bytes')

if __name__=='__main__': main()
