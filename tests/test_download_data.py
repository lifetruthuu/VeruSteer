"""Distribution checks: complete restore, damaged shards, and archive boundaries."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_data import restore


def fixture(root, name='data/example.jsonl', content=b'{"task": "example"}\n'):
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode='w') as tar:
        entry = tarfile.TarInfo(name)
        entry.size = len(content)
        tar.addfile(entry, io.BytesIO(content))
    compressed = gzip.compress(tar_bytes.getvalue(), mtime=0)
    parts = []
    for start in range(0, len(compressed), 17):
        part = root / f'{start}.bin'
        part.write_bytes(compressed[start:start+17])
        parts.append(part)
    package = dict(name='corpus', roots=['data'], compressed_bytes=len(compressed),
                   file_count=1, unpacked_bytes=len(content))
    return package, parts


class Downloads(unittest.TestCase):
    def test_split_roundtrip_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, parts = fixture(root)
            for _ in range(2):
                restore(package, parts, root / 'output')
            self.assertEqual(json.loads((root / 'output/data/example.jsonl').read_text()), {'task': 'example'})
            self.assertFalse(list((root / 'output').glob('.restore-*')))

    def test_corrupt_gzip_installs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, parts = fixture(root)
            last = bytearray(parts[-1].read_bytes())
            last[-1] ^= 1
            parts[-1].write_bytes(last)
            with self.assertRaises((OSError, EOFError)):
                restore(package, parts, root / 'output')
            self.assertFalse((root / 'output/data').exists())

    def test_archive_cannot_escape_data_root(self):
        for name in ['../outside.txt', '/absolute.txt', 'scripts/replaced.py']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package, parts = fixture(root, name)
                with self.assertRaises(ValueError):
                    restore(package, parts, root / 'output')
                self.assertFalse((root / 'outside.txt').exists())

    def test_existing_changes_require_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, parts = fixture(root)
            target = root / 'output/data/example.jsonl'
            target.parent.mkdir(parents=True)
            target.write_text('local changes')
            with self.assertRaisesRegex(ValueError, 'Existing file differs'):
                restore(package, parts, root / 'output')
            self.assertEqual(target.read_text(), 'local changes')
            restore(package, parts, root / 'output', overwrite=True)
            self.assertEqual(json.loads(target.read_text()), {'task': 'example'})


if __name__ == '__main__':
    unittest.main()
