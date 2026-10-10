"""CPU-only provenance checks for the startup color cue; no RKNN hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from rk_vision import reid
from rk_vision.pipeline import RKNNVisionPipeline
from rk_vision.reid import OSNetConfig, OSNetRKNNExtractor
from rk_vision.tracker import DeepSortTracker, DeepSortTrackerConfig
from rk_vision.yolo11 import Detection


BOX = (200., 40., 400., 460.)
FEATURE = np.array([1., 0., 0.], dtype=np.float32)
COLOR = np.ones(16, dtype=np.float32) / 4.


def extractor(**options):
    result = OSNetRKNNExtractor(OSNetConfig(
        model_path='', partial_appearance_enable=False, **options))
    result.session = SimpleNamespace(inference=lambda _: [FEATURE.copy()])
    return result


def image():
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[:] = [15, 60, 200]
    return pixels


def test_extractor_reuses_existing_color_pass_and_full_fusion(monkeypatch):
    ext = extractor()
    crops, calls = [], []
    real_signature = reid._color_signature

    def signature(crop):
        crops.append(crop.copy())
        return real_signature(crop)

    ext.session.inference = lambda _: calls.append(1) or [FEATURE.copy()]
    monkeypatch.setattr(reid, '_color_signature', signature)
    full = ext.extract(image(), [Detection(BOX, .94, 0)])
    assert len(crops) == len(calls) == 1
    expected = real_signature(image()[40:460, 200:400])
    np.testing.assert_allclose(ext.last_color_features[0], expected, atol=1e-7)
    np.testing.assert_allclose(full[0], reid._fuse_appearance_features(
        FEATURE.copy(), expected.copy(), ext.config.color_fusion_weight))


def test_rgb_color_evidence_uses_same_bgr_histogram():
    ext = extractor()
    first = ext.extract(image(), [Detection(BOX, .94, 0)], 'BGR')[0]
    color = ext.last_color_features[0].copy()
    second = ext.extract(image()[:, :, ::-1], [Detection(BOX, .94, 0)], 'RGB')[0]
    np.testing.assert_allclose(ext.last_color_features[0], color)
    np.testing.assert_allclose(first, second)


def test_color_fusion_disabled_does_not_compute_color(monkeypatch):
    ext = extractor(color_fusion_enable=False)

    def forbidden(_):
        raise AssertionError('disabled color fusion must not add computation')

    monkeypatch.setattr(reid, '_color_signature', forbidden)
    assert ext.extract(image(), [Detection(BOX, .94, 0)])[0].shape == FEATURE.shape
    assert ext.last_color_features == [None]


@pytest.mark.parametrize('failure', ['empty', 'disabled', 'no_session', 'no_output',
                                      'exception', 'bad_frame'])
def test_each_failed_capture_clears_previous_color(failure):
    ext = extractor()
    detections = [Detection(BOX, .94, 0)]
    ext.extract(image(), detections)
    assert ext.last_color_features[0] is not None
    if failure == 'empty':
        ext.extract(image(), [])
        assert ext.last_color_features == []
        return
    if failure == 'disabled':
        ext.config = OSNetConfig(model_path='', enabled=False)
    elif failure == 'no_session':
        ext.session = None
    elif failure == 'no_output':
        ext.session.inference = lambda _: []
    elif failure == 'exception':
        def fail(_):
            raise RuntimeError('synthetic inference failure')
        ext.session.inference = fail
        with pytest.raises(RuntimeError):
            ext.extract(image(), detections)
        assert ext.last_color_features == [None]
        return
    elif failure == 'bad_frame':
        with pytest.raises(ValueError):
            ext.extract(np.zeros((2, 2)), detections)
        assert ext.last_color_features == [None]
        return
    ext.extract(image(), detections)
    assert ext.last_color_features == [None]


def test_invalid_crop_keeps_color_aligned_with_detection_index():
    ext = extractor()
    ext.extract(image(), [Detection((20, 20, 20, 40), .95, 0), Detection(BOX, .94, 0)])
    assert ext.last_color_features[0] is None
    assert len(ext.last_color_features) == 2
    assert ext.last_color_features[1] is not None


def tracker():
    return DeepSortTracker(DeepSortTrackerConfig(
        n_init=1, identity_appearance_region_safety_enable=True))


def update(t, frame, colors=None, detections=None, features=None, **context):
    if detections is None:
        detections = [Detection(BOX, .94, 0)]
    if features is None:
        features = [FEATURE for _ in detections]
    return t.update(detections, features, color_features=colors,
        image_width=640, image_height=480, frame_context=dict(
            control_frame_id=frame, capture_frame_id=100 + frame,
            capture_timestamp=10. + frame * .1, **context))


def test_current_color_binds_to_source_box_after_detection_reordering():
    t = tracker()
    left, right = Detection((80, 40, 240, 460), .94, 0), Detection((390, 40, 560, 460), .95, 0)
    left_full, right_full = FEATURE, np.array([0., 1., 0.], dtype=np.float32)
    left_color, right_color = np.eye(16)[2], np.eye(16)[9]
    update(t, 1, [left_color, right_color], [left, right], [left_full, right_full])
    update(t, 2, [right_color, left_color], [right, left], [right_full, left_full])
    assert len(t.last_identity_observations) == 2
    for observation in t.last_identity_observations:
        metadata = observation['sample_metadata']
        source = metadata['source_detection_index']
        assert metadata['initial_color_feature'] == [right_color, left_color][source].tolist()
        assert metadata['initial_color_source'] == 'hsv_crop_v1_bgr'
        assert metadata['capture_frame_id'] == 102
        assert metadata['capture_timestamp'] == pytest.approx(10.2)
        assert metadata['track_id'] == observation['raw_track_id']
        assert tuple(metadata['detector_bbox']) == [right, left][source].bbox


@pytest.mark.parametrize('colors', [None, [], [COLOR, COLOR], [None], [np.ones(6)],
                                      [np.zeros(16)], [np.full(16, np.nan)],
                                      [np.full(16, np.inf)], [np.full(16, -1.)]])
def test_missing_misaligned_or_invalid_color_never_reuses_previous(colors):
    t = tracker()
    update(t, 1, [COLOR])
    update(t, 2, [COLOR])
    assert 'initial_color_feature' in t.last_identity_observations[0]['sample_metadata']
    update(t, 3, colors, initial_color_feature=COLOR.tolist(),
           initial_color_source='hsv_crop_v1_bgr')
    metadata = t.last_identity_observations[0]['sample_metadata']
    assert 'initial_color_feature' not in metadata
    assert 'initial_color_source' not in metadata


def test_predicted_observation_cannot_inherit_a_color(monkeypatch):
    t = tracker()
    update(t, 1, [COLOR])
    update(t, 2, [COLOR])
    captured = []
    assign = t.identity_bank.assign

    def capture(**kwargs):
        captured.append(kwargs['sample_metadata'])
        return assign(**kwargs)

    monkeypatch.setattr(t.identity_bank, 'assign', capture)
    update(t, 3, [COLOR], [], [], initial_color_feature=COLOR.tolist(),
           initial_color_source='hsv_crop_v1_bgr')
    assert captured and not captured[0]['is_fresh']
    assert all('initial_color_feature' not in m for m in captured)
    assert not t.last_identity_observations


def test_color_stops_at_completed_initial_enrollment():
    t = tracker()
    for frame in (1, 2, 3):
        update(t, frame, [COLOR])
    assert t.identity_bank.identities
    assert 'initial_color_feature' in t.last_identity_observations[0]['sample_metadata']
    update(t, 4, [COLOR])
    assert 'initial_color_feature' not in t.last_identity_observations[0]['sample_metadata']


@pytest.mark.parametrize('kind', ['current', 'legacy', 'wrong_length'])
def test_pipeline_forwards_only_aligned_color_and_supports_legacy_extractors(kind):
    calls = []
    p = RKNNVisionPipeline.__new__(RKNNVisionPipeline)
    p.config = SimpleNamespace(predicted_reid_verify_enable=False,
                               identity_suppress_duplicate_uids=False,
                               person_class_id=0, conf_threshold=.35)
    p.logger = None
    p._frame_context = {'capture_frame_id': 17, 'capture_timestamp': 10.1}
    p.detector = SimpleNamespace(detect=lambda *_: [Detection(BOX, .94, 0)], last_timing_ms={})
    p.reid = SimpleNamespace(extract=lambda *_: [FEATURE], last_timing_ms={})
    if kind != 'legacy':
        p.reid.last_color_features = [COLOR] if kind == 'current' else [COLOR, COLOR]
    p.tracker = SimpleNamespace(update=lambda *args, **kwargs: calls.append(kwargs) or [])
    # This test stubs detector bookkeeping, so provide the current empty
    # diagnostic list that the real _record_detector_output publishes.
    p.last_search_diagnostic_detections = []
    p._record_detector_output = lambda detections: detections
    p._record_reid_diagnostics = lambda *_: None
    assert p.process_frame(image()) == []
    assert len(calls) == 1
    if kind == 'current':
        np.testing.assert_array_equal(calls[0]['color_features'][0], COLOR)
    else:
        assert 'color_features' not in calls[0]
    assert calls[0]['frame_context'] == p._frame_context
