from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .frames import numpy_from_frame
from .runtime import RKNNInferenceSession


@dataclass(frozen=True)
class Detection:
    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class YOLO11Config:
    model_path: str
    input_size: int = 640
    conf_threshold: float = 0.25
    search_diagnostic_conf_threshold: float = 0.10
    search_diagnostic_class_id: int = 0
    nms_threshold: float = 0.45
    num_classes: int = 80
    input_format: str = "RGB"
    output_box_format: str = "xywh"
    target: str = "rk3588"
    core_mask: str = "auto"
    backend: str = "auto"


@dataclass(frozen=True)
class LetterboxInfo:
    src_width: int
    src_height: int
    input_size: int
    scale: float
    pad_x: float
    pad_y: float


class YOLO11RKNNDetector:
    def __init__(self, config: YOLO11Config) -> None:
        self.config = config
        self.last_timing_ms = {
            "preprocess": 0.0,
            "inference": 0.0,
            "decode": 0.0,
            "nms": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }
        self.search_diagnostic_active = False
        self.last_search_diagnostic_detections: List[Detection] = []
        self.session = RKNNInferenceSession(
            config.model_path,
            target=config.target,
            core_mask=config.core_mask,
            backend=config.backend,
        )

    def detect(self, frame: Any, frame_format: str = "BGR") -> List[Detection]:
        start = time.perf_counter()
        prep_start = start
        tensor, letterbox = prepare_yolo_input(frame, self.config, frame_format)
        infer_start = time.perf_counter()
        outputs = self.session.inference([tensor], data_format=["nhwc"])
        post_start = time.perf_counter()
        diagnostic_threshold = max(
            0.01,
            min(float(self.config.conf_threshold), float(self.config.search_diagnostic_conf_threshold)),
        )
        decode_config = self.config
        if self.search_diagnostic_active and diagnostic_threshold < float(self.config.conf_threshold):
            decode_config = replace(self.config, conf_threshold=diagnostic_threshold)
        decoded_detections = _decode_yolo11_outputs(outputs, decode_config, letterbox)
        if self.search_diagnostic_active:
            self.last_search_diagnostic_detections = [
                detection
                for detection in decoded_detections
                if int(detection.class_id) == int(self.config.search_diagnostic_class_id)
            ]
        else:
            self.last_search_diagnostic_detections = []
        # The low-threshold search probe is diagnostic only. The production
        # detector, tracker, ReID, and motor control keep the configured gate.
        raw_detections = [
            detection
            for detection in decoded_detections
            if float(detection.score) >= float(self.config.conf_threshold)
        ]
        nms_start = time.perf_counter()
        detections = _nms(raw_detections, self.config.nms_threshold)
        end = time.perf_counter()
        self.last_timing_ms = {
            "preprocess": _elapsed_ms(prep_start, infer_start),
            "inference": _elapsed_ms(infer_start, post_start),
            "decode": _elapsed_ms(post_start, nms_start),
            "nms": _elapsed_ms(nms_start, end),
            "postprocess": _elapsed_ms(post_start, end),
            "total": _elapsed_ms(start, end),
        }
        return detections

    def set_search_diagnostic_active(self, active: bool) -> None:
        self.search_diagnostic_active = bool(active)
        if not self.search_diagnostic_active:
            self.last_search_diagnostic_detections = []

    def load(self) -> None:
        self.session.load()

    def release(self) -> None:
        self.session.release()


def prepare_yolo_input(frame: Any, config: YOLO11Config, frame_format: str = "BGR"):
    np = _np()
    arr, width, height, fmt = numpy_from_frame(frame, frame_format)
    img, info = _letterbox(arr, width, height, config.input_size)
    desired = config.input_format.upper()
    if fmt != desired:
        if {fmt, desired} == {"BGR", "RGB"}:
            img = img[:, :, ::-1]
        else:
            raise ValueError(f"cannot convert frame format {fmt!r} to model format {desired!r}")
    tensor = np.expand_dims(np.ascontiguousarray(img), 0)
    return tensor, info


def postprocess_yolo11_outputs(
    outputs: Sequence[Any],
    config: YOLO11Config,
    letterbox: LetterboxInfo,
) -> List[Detection]:
    raw = _decode_yolo11_outputs(outputs, config, letterbox)
    return _nms(raw, config.nms_threshold)


def _decode_yolo11_outputs(
    outputs: Sequence[Any],
    config: YOLO11Config,
    letterbox: LetterboxInfo,
) -> List[Detection]:
    """Decode RKNN outputs without NMS so detector timing can measure both stages."""
    np = _np()
    if not outputs:
        return []
    arrays = [np.asarray(out) for out in outputs]
    raw = _postprocess_feature_outputs(arrays, config, letterbox)
    if raw is None:
        raw = _postprocess_single_output(arrays[0], config, letterbox)
    return raw


def _postprocess_feature_outputs(
    outputs: Sequence[Any],
    config: YOLO11Config,
    letterbox: LetterboxInfo,
) -> Optional[List[Detection]]:
    np = _np()
    if len(outputs) < 6 or len(outputs) % 3 != 0:
        return None

    pair_per_branch = len(outputs) // 3
    boxes_all = []
    scores_all = []
    classes_all = []
    for i in range(3):
        box_raw = np.asarray(outputs[pair_per_branch * i])
        cls_raw = np.asarray(outputs[pair_per_branch * i + 1])
        if box_raw.ndim != 4 or cls_raw.ndim != 4:
            return None
        if box_raw.shape[1] % 4 != 0 and box_raw.shape[-1] % 4 == 0:
            box_raw = box_raw.transpose(0, 3, 1, 2)
        if cls_raw.shape[1] != config.num_classes and cls_raw.shape[-1] == config.num_classes:
            cls_raw = cls_raw.transpose(0, 3, 1, 2)
        if box_raw.shape[1] % 4 != 0:
            return None
        boxes = _box_process(box_raw, config.input_size)
        scores = _sigmoid(cls_raw) if cls_raw.max() > 1.0 or cls_raw.min() < 0.0 else cls_raw
        boxes = boxes.reshape(4, -1).T
        scores = scores.reshape(scores.shape[1], -1).T
        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = confs >= config.conf_threshold
        if keep.any():
            boxes_all.append(boxes[keep])
            scores_all.append(confs[keep])
            classes_all.append(class_ids[keep])

    if not boxes_all:
        return []
    boxes_np = np.concatenate(boxes_all, axis=0)
    scores_np = np.concatenate(scores_all, axis=0)
    classes_np = np.concatenate(classes_all, axis=0)
    return [
        Detection(_scale_box_to_source(tuple(map(float, box)), letterbox), float(score), int(cls))
        for box, score, cls in zip(boxes_np, scores_np, classes_np)
    ]


def _postprocess_single_output(output: Any, config: YOLO11Config, letterbox: LetterboxInfo) -> List[Detection]:
    np = _np()
    pred = np.asarray(output)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"unsupported YOLO11 output shape: {np.asarray(output).shape!r}")
    channel_first_scores = pred.shape[0] == 4 + config.num_classes and pred.shape[1] != 6
    if channel_first_scores or (
        pred.shape[0] in {5 + config.num_classes, 4 + config.num_classes, 6}
        and pred.shape[0] < pred.shape[1]
    ):
        pred = pred.T

    detections: List[Detection] = []
    cols = pred.shape[1]
    if cols == 6 and not channel_first_scores:
        boxes = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(int)
    elif cols >= 5 + config.num_classes:
        boxes = pred[:, :4]
        obj = pred[:, 4]
        cls_scores = pred[:, 5 : 5 + config.num_classes]
        class_ids = cls_scores.argmax(axis=1)
        scores = obj * cls_scores.max(axis=1)
    elif cols >= 4 + config.num_classes:
        boxes = pred[:, :4]
        cls_scores = pred[:, 4 : 4 + config.num_classes]
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1)
    else:
        raise ValueError(f"unsupported YOLO11 output columns: {cols}")

    max_score = float(scores.max()) if scores.size else 0.0
    min_score = float(scores.min()) if scores.size else 0.0
    if max_score > 1.0 or min_score < 0.0:
        scores = _sigmoid(scores)
    max_box = float(boxes.max()) if boxes.size else 0.0
    normalized = max_box <= 2.0
    if normalized:
        boxes = boxes * float(config.input_size)

    for box, score, class_id in zip(boxes, scores, class_ids):
        score_f = float(score)
        if score_f < config.conf_threshold:
            continue
        x1, y1, x2, y2 = _box_to_xyxy(tuple(map(float, box)), config.output_box_format)
        detections.append(
            Detection(
                _scale_box_to_source((x1, y1, x2, y2), letterbox),
                score_f,
                int(class_id),
            )
        )
    return detections


def _box_process(position: Any, input_size: int):
    np = _np()
    n, c, h, w = position.shape
    reg_max = c // 4
    position = position.reshape(n, 4, reg_max, h, w)
    position = position - position.max(axis=2, keepdims=True)
    prob = np.exp(position)
    prob = prob / prob.sum(axis=2, keepdims=True)
    acc = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    position = (prob * acc).sum(axis=2)[0]

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=0).astype(np.float32)
    stride = float(input_size) / float(h)
    x1y1 = (grid + 0.5 - position[0:2]) * stride
    x2y2 = (grid + 0.5 + position[2:4]) * stride
    return np.concatenate((x1y1, x2y2), axis=0)


def _letterbox(img: Any, width: int, height: int, input_size: int):
    np = _np()
    scale = min(float(input_size) / float(width), float(input_size) / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = _resize(img, new_w, new_h)
    pad_x = (input_size - new_w) / 2.0
    pad_y = (input_size - new_h) / 2.0
    canvas = np.zeros((input_size, input_size, 3), dtype=resized.dtype)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + new_h, left : left + new_w, :] = resized
    return canvas, LetterboxInfo(width, height, input_size, scale, float(left), float(top))


def _resize(img: Any, width: int, height: int):
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        np = _np()
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("OpenCV or Pillow is required for rk_vision image resizing") from exc
        return np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def _box_to_xyxy(box: Tuple[float, float, float, float], fmt: str) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    if fmt.lower() == "xyxy":
        return x, y, w, h
    if fmt.lower() != "xywh":
        raise ValueError(f"unsupported YOLO box format: {fmt!r}")
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def _scale_box_to_source(box: Tuple[float, float, float, float], info: LetterboxInfo) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = (x1 - info.pad_x) / info.scale
    x2 = (x2 - info.pad_x) / info.scale
    y1 = (y1 - info.pad_y) / info.scale
    y2 = (y2 - info.pad_y) / info.scale
    return (
        max(0.0, min(float(info.src_width), x1)),
        max(0.0, min(float(info.src_height), y1)),
        max(0.0, min(float(info.src_width), x2)),
        max(0.0, min(float(info.src_height), y2)),
    )


def _nms(dets: Iterable[Detection], threshold: float) -> List[Detection]:
    det_list = sorted(dets, key=lambda d: d.score, reverse=True)
    cv2_keep = _cv2_nms(det_list, threshold)
    if cv2_keep is not None:
        return cv2_keep
    keep: List[Detection] = []
    while det_list:
        best = det_list.pop(0)
        keep.append(best)
        det_list = [
            det
            for det in det_list
            if det.class_id != best.class_id or _iou(best.bbox, det.bbox) <= threshold
        ]
    return keep


def _cv2_nms(dets: List[Detection], threshold: float) -> Optional[List[Detection]]:
    if not dets:
        return []
    try:
        import cv2
    except Exception:
        return None
    keep: List[Detection] = []
    class_ids = sorted({det.class_id for det in dets})
    for class_id in class_ids:
        cls_dets = [det for det in dets if det.class_id == class_id]
        boxes = []
        scores = []
        for det in cls_dets:
            x1, y1, x2, y2 = det.bbox
            boxes.append([float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))])
            scores.append(float(det.score))
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=float(threshold))
        if len(indices) == 0:
            continue
        for index in _np().asarray(indices).reshape(-1):
            keep.append(cls_dets[int(index)])
    return sorted(keep, key=lambda det: det.score, reverse=True)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _sigmoid(x: Any):
    np = _np()
    return 1.0 / (1.0 + np.exp(-x))


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for YOLO11 RKNN postprocess") from exc
    return np
