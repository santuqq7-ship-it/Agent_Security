#!/usr/bin/env python3
"""Download and verify the official ToolSafe TS-Bench files.

The upstream repository stores these JSON files as Git LFS pointers.  The
normal ``git lfs pull`` endpoint currently reports an exhausted LFS budget, so
this script reads the same public objects through GitHub's media endpoint and
verifies every download against the SHA-256 and byte size recorded in the
pointer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "data" / "TS-Bench"
MEDIA_ROOT = "https://media.githubusercontent.com/media/MurrayTom/ToolSafe/main/TS-Bench"

# These values are copied from the upstream Git LFS pointer files.  Keeping
# them here makes a download reproducible even when the LFS service is down.
FILES = {
    "agentdojo-traj/banking.json": ("eb4a314845c8236ba1bbcd9855973f59279a70368b6eb31d447a4cd243086029", 239239),
    "agentdojo-traj/slack.json": ("08152de8cf9c39d01d58eb708f3a820a892f493841b975e09654ab904cf8a0d5", 361979),
    "agentdojo-traj/travel.json": ("1935121585fb1a02662e8121fe80e78c6d89b77e0c6035ea2b75b593e3728a22", 1071254),
    "agentdojo-traj/workspace.json": ("74b20a4fcd7df1c3e39a231e64860aeac8cd57ce3531728b329080abbcc862d0", 8959525),
    "agentharm-traj/benign_steps.json": ("f44fce25b0ed795e3c11fadafb8d8c959d2ae4513e6e9f15e154fa4aba9bf020", 689251),
    "agentharm-traj/harmful_steps.json": ("6046b4f4b2e5086c98780433bddf74be542eeec910e3260b4727bf22a9a3365b", 1764105),
    "asb-traj/test/DPI_attack_success.json": ("a900a0775addf35588154d28438127f9c830a08845a105e084b319b1e984d4af", 5205776),
    "asb-traj/test/OPI_attack_success.json": ("e491c96e43c8352e65ca948a72cb3e6f366d60f14fd7d3d6e374865eb092ae47", 4712262),
    "asb-traj/test/atttack_failure.json": ("6cf422389a41416e05d236252ecba8757c1f7d7a978ae270e419ff846ae98f51", 945992),
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading the whole file in RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_hash: str, expected_size: int) -> None:
    """Fail loudly unless size, hash, and JSON syntax all match the manifest."""

    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    if actual_size != expected_size or actual_hash != expected_hash:
        raise RuntimeError(
            f"verification failed for {path}: "
            f"size {actual_size}/{expected_size}, hash {actual_hash}/{expected_hash}"
        )

    # Parsing catches truncated or otherwise invalid downloads in addition to
    # the cryptographic check.
    with path.open("r", encoding="utf-8") as stream:
        json.load(stream)


def fetch(output_dir: Path, force: bool) -> None:
    """Download each object atomically and verify it before exposing the file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")
    print(f"source: {MEDIA_ROOT}")

    for relative_path, (expected_hash, expected_size) in FILES.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not force:
            verify(destination, expected_hash, expected_size)
            print(f"verified existing: {relative_path}")
            continue

        partial = destination.with_suffix(destination.suffix + ".part")
        url = f"{MEDIA_ROOT}/{relative_path}"
        print(f"downloading: {relative_path}")
        request = urllib.request.Request(url, headers={"User-Agent": "ToolSafe-TS-Bench-fetcher/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)

        verify(partial, expected_hash, expected_size)
        partial.replace(destination)
        print(f"downloaded and verified: {relative_path}")

    print(f"verified {len(FILES)} TS-Bench files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="destination directory; defaults to practice/toolsafe_reproduction/data/TS-Bench",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload files even when an existing file already passes verification",
    )
    args = parser.parse_args()
    fetch(args.output_dir.expanduser().resolve(), args.force)


if __name__ == "__main__":
    main()
