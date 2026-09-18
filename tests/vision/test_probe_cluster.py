from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rk_vision.pipeline import cluster_probe_detections
from rk_vision.yolo11 import Detection


def test_overlapping_probe_fragments_collapse_to_one_representative() -> None:
    probes = (
        Detection((14.9, 5.1, 317.8, 474.7), 0.1921, 0),
        Detection((12.3, 5.3, 309.0, 474.5), 0.1808, 0),
        Detection((17.3, 4.1, 405.6, 478.8), 0.1771, 0),
        Detection((39.9, 3.9, 410.1, 478.7), 0.1695, 0),
    )

    representatives, diagnostics = cluster_probe_detections(probes)

    assert len(representatives) == 1
    assert representatives[0] == probes[0]
    assert len(diagnostics) == 1
    assert diagnostics[0]["member_count"] == 4
    assert diagnostics[0]["ambiguous"] is False


def test_competing_probe_clusters_with_small_score_gap_are_rejected() -> None:
    probes = (
        Detection((20.0, 20.0, 130.0, 300.0), 0.192, 0),
        Detection((24.0, 24.0, 128.0, 298.0), 0.180, 0),
        Detection((500.0, 30.0, 620.0, 310.0), 0.184, 0),
        Detection((504.0, 32.0, 618.0, 308.0), 0.173, 0),
    )

    representatives, diagnostics = cluster_probe_detections(
        probes,
        min_score_gap=0.03,
    )

    assert representatives == ()
    assert len(diagnostics) == 2
    assert diagnostics[0]["member_count"] == 2
    assert diagnostics[1]["member_count"] == 2
    assert diagnostics[0]["ambiguous"] is True


def test_competing_probe_clusters_with_clear_score_gap_keep_top_cluster() -> None:
    probes = (
        Detection((20.0, 20.0, 130.0, 300.0), 0.205, 0),
        Detection((24.0, 24.0, 128.0, 298.0), 0.180, 0),
        Detection((500.0, 30.0, 620.0, 310.0), 0.151, 0),
    )

    representatives, diagnostics = cluster_probe_detections(
        probes,
        min_score_gap=0.03,
    )

    assert len(representatives) == 1
    assert representatives[0] == probes[0]
    assert len(diagnostics) == 2
    assert diagnostics[0]["score_gap"] == pytest.approx(0.054)
    assert diagnostics[0]["ambiguous"] is False
