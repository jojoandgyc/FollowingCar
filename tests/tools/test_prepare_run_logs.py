#!/usr/bin/env python3
"""Focused tests for launcher run-directory retention."""

from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.prepare_run_logs import (  # noqa: E402
    MANAGED_MARKER,
    MANAGED_OWNER,
    MARKER_VERSION,
    prepare_run_directory,
)


class PrepareRunLogsTests(unittest.TestCase):
    def test_keeps_latest_three_and_preserves_unmanaged_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_dir = root / "run_user_notes"
            user_dir.mkdir()
            (user_dir / "keep.txt").write_text("keep", encoding="ascii")
            user_link = root / "run_link"
            user_link.symlink_to(user_dir, target_is_directory=True)

            created = [prepare_run_directory(root, migrate_legacy=False) for _ in range(5)]
            managed = [
                entry
                for entry in root.iterdir()
                if (entry / MANAGED_MARKER).is_file()
            ]
            self.assertEqual(len(managed), 3)
            self.assertTrue(created[-1].is_dir())
            self.assertTrue(user_dir.exists())
            self.assertTrue(user_link.is_symlink())

    def test_migrates_only_known_flat_records_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "request_0513_modular.log").write_text("log", encoding="ascii")
            (root / "camera_raw.avi").write_bytes(b"avi")
            (root / "camera_raw.frames.csv").write_text("csv", encoding="ascii")
            search_frames = root / "search_frames"
            search_frames.mkdir()
            (search_frames / "frame.jpg").write_bytes(b"jpg")
            (root / "notes.txt").write_text("keep", encoding="ascii")

            current = prepare_run_directory(root, keep=3)
            legacy = [entry for entry in root.iterdir() if entry.name.startswith("run_legacy_")]
            self.assertEqual(len(legacy), 1)
            for name in ("request_0513_modular.log", "camera_raw.avi", "camera_raw.frames.csv", "search_frames"):
                self.assertFalse((root / name).exists())
                self.assertTrue((legacy[0] / name).exists())
            self.assertTrue((root / "notes.txt").exists())
            marker = json.loads((current / MANAGED_MARKER).read_text(encoding="ascii"))
            self.assertEqual(marker["owner"], MANAGED_OWNER)
            prepare_run_directory(root, keep=3)
            legacy_after = [entry for entry in root.iterdir() if entry.name.startswith("run_legacy_")]
            self.assertEqual(legacy_after, legacy)
            prepare_run_directory(root, keep=3)
            self.assertFalse(legacy[0].exists())
            self.assertEqual(sum(entry.is_dir() for entry in root.iterdir()), 3)

    def test_protects_symlink_markers_and_legacy_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker_source = root / "user_marker"
            marker_source.write_text(
                json.dumps({"owner": MANAGED_OWNER, "version": MARKER_VERSION, "sequence": 1}),
                encoding="ascii",
            )
            user_dir = root / "run_user"
            user_dir.mkdir()
            (user_dir / MANAGED_MARKER).symlink_to(marker_source)
            (root / "camera_raw.avi").symlink_to(marker_source)
            wrong_type = root / "request_0513_modular.log"
            wrong_type.mkdir()

            for _ in range(4):
                prepare_run_directory(root)
            self.assertTrue(user_dir.is_dir())
            self.assertTrue((user_dir / MANAGED_MARKER).is_symlink())
            self.assertTrue((root / "camera_raw.avi").is_symlink())
            self.assertTrue(wrong_type.is_dir())
            self.assertTrue(marker_source.is_file())

    def test_rejects_invalid_retention_without_changing_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "logs"
            with self.assertRaises(ValueError):
                prepare_run_directory(root, keep=0)
            self.assertFalse(root.exists())

    def test_same_second_runs_have_unique_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = prepare_run_directory(root, migrate_legacy=False)
            second = prepare_run_directory(root, migrate_legacy=False)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_creation_sequence_ignores_frozen_clock_uuid_order_and_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixed_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
            with mock.patch("tools.prepare_run_logs._datetime.datetime") as clock:
                clock.now.return_value = fixed_time
                created = []
                for suffix in ("ffffffff", "cccccccc", "aaaaaaaa", "11111111", "00000000"):
                    with mock.patch("tools.prepare_run_logs.uuid.uuid4") as uuid_mock:
                        uuid_mock.return_value.hex = suffix
                        created.append(prepare_run_directory(root, migrate_legacy=False))
                    # An old recording receiving a late artifact must not
                    # become newer than a later invocation.
                    if created[0].exists():
                        os.utime(created[0], (9999999999, 9999999999))
                        os.utime(created[0] / MANAGED_MARKER, (9999999999, 9999999999))
            remaining = {entry for entry in root.iterdir() if entry.is_dir()}
            self.assertEqual(remaining, set(created[-3:]))

    def test_symlink_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            real_root = base / "real"
            real_root.mkdir()
            link_root = base / "link"
            link_root.symlink_to(real_root, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                prepare_run_directory(link_root)


if __name__ == "__main__":
    unittest.main()
