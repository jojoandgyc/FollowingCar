#!/usr/bin/env python3
"""Prepare an isolated run directory and retain the latest run records.

The launcher owns one stable log root.  This helper creates a marked child
directory for each invocation, migrates only the known legacy flat records,
and removes only older marked child directories once the retention limit is
exceeded.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import fcntl
import json
import os
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


MANAGED_MARKER = ".run_managed"
MANAGED_OWNER = "rk_car_runtime_module"
MARKER_VERSION = 1
PREPARE_LOCK = ".run_prepare.lock"
RUN_PREFIX = "run_"
# These are the artifacts produced directly in the historical flat log root.
# Do not broaden this list: arbitrary user files must remain untouched.
LEGACY_RECORD_NAMES = (
    "request_0513_modular.log",
    "camera_raw.avi",
    "camera_raw.frames.csv",
    "search_frames",
    "peripheral_preflight.log",
    "force_motor_safe_stop.log",
)


def _new_run_name(*, legacy: bool = False) -> str:
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    prefix = "run_legacy" if legacy else RUN_PREFIX.rstrip("_")
    return f"{prefix}_{timestamp}_{suffix}"


def _write_marker(directory: Path, *, sequence: int) -> None:
    marker = directory / MANAGED_MARKER
    # Exclusive creation avoids silently accepting a pre-existing user file.
    with marker.open("x", encoding="ascii") as stream:
        json.dump(
            {"owner": MANAGED_OWNER, "version": MARKER_VERSION, "sequence": sequence},
            stream,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _create_managed_directory(root: Path, *, sequence: int, legacy: bool = False) -> Path:
    for _ in range(20):
        directory = root / _new_run_name(legacy=legacy)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        _write_marker(directory, sequence=sequence)
        return directory
    raise RuntimeError("could not allocate a unique run directory")


def _is_real_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _marker_sequence(path: Path) -> Optional[int]:
    if not _is_real_directory(path) or not path.name.startswith(RUN_PREFIX):
        return None
    marker = path / MANAGED_MARKER
    if marker.is_symlink() or not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    sequence = data.get("sequence")
    if (
        data.get("owner") != MANAGED_OWNER
        or data.get("version") != MARKER_VERSION
        or type(sequence) is not int
        or sequence < 1
    ):
        return None
    return sequence


def _managed_directories(root: Path) -> list[tuple[int, Path]]:
    entries: list[tuple[int, Path]] = []
    for entry in root.iterdir():
        sequence = _marker_sequence(entry)
        if sequence is not None:
            entries.append((sequence, entry))
    # Filesystem timestamps may be coarse or changed when runtime writes
    # artifacts.  The immutable sequence records creation order instead.
    entries.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return entries


def _legacy_records(root: Path) -> list[Path]:
    records: list[Path] = []
    for name in LEGACY_RECORD_NAMES:
        entry = root / name
        # Symlinks are deliberately not migrated.  This avoids moving a link
        # that points outside the log root and leaves it available to the user.
        if entry.is_symlink():
            continue
        if (name == "search_frames" and entry.is_dir()) or (
            name != "search_frames" and entry.is_file()
        ):
            records.append(entry)
    return records


def _migrate_legacy_records(root: Path, *, sequence: int) -> Optional[Path]:
    records = _legacy_records(root)
    if not records:
        return None
    directory = _create_managed_directory(root, sequence=sequence, legacy=True)
    for entry in records:
        shutil.move(str(entry), str(directory / entry.name))
    return directory


def _validate_root(root: Path) -> Path:
    if root == Path(root.anchor):
        raise RuntimeError(f"refusing filesystem root as log root: {root}")
    if root.exists() and root.is_symlink():
        raise RuntimeError(f"refusing symlink log root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"log root is not a real directory: {root}")
    return root


def _prune(root: Path, *, keep: int, current: Path) -> None:
    previous = [item for _, item in _managed_directories(root) if item != current]
    for directory in previous[keep - 1:]:
        # Re-check the direct-child boundary before deleting anything.  The
        # marker and real-directory checks above are intentionally repeated.
        if directory == current or directory.parent != root or _marker_sequence(directory) is None:
            continue
        shutil.rmtree(directory)


@contextmanager
def _preparation_lock(root: Path):
    # This lock covers only directory preparation, never vehicle operation.
    # Refuse a symlink lock file and a concurrent preparation attempt.
    descriptor = os.open(root / PREPARE_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def prepare_run_directory(
    root: str | os.PathLike[str],
    *,
    keep: int = 3,
    migrate_legacy: bool = True,
) -> Path:
    """Create a managed run directory under *root* and retain at most *keep*.

    The returned path is absolute.  Only immediate, real child directories
    carrying this helper's marker are eligible for pruning.
    """

    if keep < 1:
        raise ValueError("keep must be at least 1")
    root_path = _validate_root(Path(root).expanduser().absolute())
    with _preparation_lock(root_path):
        managed = _managed_directories(root_path)
        sequence = max((item[0] for item in managed), default=0) + 1
        if migrate_legacy:
            legacy = _migrate_legacy_records(root_path, sequence=sequence)
            if legacy is not None:
                sequence += 1
        current = _create_managed_directory(root_path, sequence=sequence)
        _prune(root_path, keep=keep, current=current)
    return current


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="stable log root directory")
    parser.add_argument("--keep", type=int, default=3, help="number of managed runs to retain")
    parser.add_argument(
        "--no-migrate-legacy",
        action="store_true",
        help="leave known flat legacy records in place",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        directory = prepare_run_directory(
            args.root,
            keep=args.keep,
            migrate_legacy=not args.no_migrate_legacy,
        )
    except Exception as exc:
        print(f"prepare run logs failed: {exc}", file=sys.stderr)
        return 1
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
