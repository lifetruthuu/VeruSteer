#!/usr/bin/env python3
"""Restore the research data from local or anonymously downloaded gzip shards."""
import argparse
import filecmp
import gzip
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_URL = "https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/"
ROOT = Path(__file__).resolve().parents[1]


def fetch(url, target, expected_bytes=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchArtifactDownloader/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise ValueError(f"Incomplete download: {target.name}")
            partial.replace(target)
            return
        except (OSError, ValueError, urllib.error.URLError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Invalid package path: {value!r}")
    return path


def restore(package, parts, destination, overwrite=False):
    """Check the entire gzip stream before installing regular files."""
    roots = [safe_relative(value) for value in package["roots"]]
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination) as tmp:
        temporary = Path(tmp)
        archive = temporary / "payload.tar.gz"
        with archive.open("wb") as out:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
        if archive.stat().st_size != package["compressed_bytes"]:
            raise ValueError("Compressed package size does not match the manifest")
        staging = temporary / "files"
        staging.mkdir()
        members = []
        seen = set()
        total = 0
        with gzip.open(archive, "rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as tar:
                for member in tar:
                    relative = safe_relative(member.name)
                    if not any(relative.is_relative_to(root) for root in roots):
                        raise ValueError(f"File outside the package roots: {relative}")
                    if not member.isfile() or relative in seen:
                        raise ValueError(f"Unsupported or duplicate archive member: {relative}")
                    seen.add(relative)
                    total += member.size
                    if total > package["unpacked_bytes"] or len(seen) > package["file_count"]:
                        raise ValueError("Archive exceeds its declared inventory")
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out, length=1024 * 1024)
                    if target.stat().st_size != member.size:
                        raise ValueError(f"Truncated member: {relative}")
                    members.append(relative)
            # Reading to EOF verifies gzip's CRC and length trailer as well.
            while compressed.read(1024 * 1024):
                pass
        if len(members) != package["file_count"] or total != package["unpacked_bytes"]:
            raise ValueError("Unpacked inventory does not match the manifest")
        pending = []
        for relative in members:
            source, target = staging / relative, destination / relative
            if any(p.is_symlink() for p in [target, *target.parents] if p != destination.parent):
                raise ValueError(f"Refusing a symlink destination: {relative}")
            if target.exists():
                if target.is_file() and filecmp.cmp(source, target, shallow=False):
                    continue
                if not overwrite or not target.is_file():
                    raise ValueError(f"Existing file differs: {relative}; use --overwrite to replace data")
            pending.append((source, target))
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
    print(f"Restored {package['name']}: {len(members)} files, {total:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=["all", "corpus", "selector"], default="all")
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--base-url", default=DEFAULT_URL,
                        help="Repository file endpoint; defaults to the anonymous mirror")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if urllib.parse.urlparse(args.base_url).scheme not in {"http", "https"}:
        parser.error("--base-url must use HTTP or HTTPS")
    cache = args.destination.resolve() / "downloads"
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        fetch(urllib.parse.urljoin(args.base_url.rstrip('/') + '/', "downloads/manifest.json"), manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "tar.gz-split-v1":
        raise ValueError("Unsupported data manifest")
    packages = manifest["packages"]
    if args.package != "all":
        packages = [p for p in packages if p["name"] == args.package]
    if not packages:
        raise ValueError("Requested package is absent from the manifest")
    for package in packages:
        parts = []
        for index, part in enumerate(package["parts"], 1):
            relative = safe_relative(part["path"])
            if len(relative.parts) != 2 or relative.parts[0] != "downloads" or not relative.name.endswith(".bin"):
                raise ValueError(f"Invalid shard path: {relative}")
            target = args.destination.resolve() / relative
            if not target.is_file() or target.stat().st_size != part["bytes"]:
                print(f"Downloading {package['name']} {index}/{len(package['parts'])}: {relative.name}", flush=True)
                url = urllib.parse.urljoin(args.base_url.rstrip('/') + '/', urllib.parse.quote(str(relative)))
                fetch(url, target, part["bytes"])
            parts.append(target)
        if not args.download_only:
            restore(package, parts, args.destination, args.overwrite)
    print("Data downloads complete." if args.download_only else "Research data are ready.")


if __name__ == "__main__":
    main()
