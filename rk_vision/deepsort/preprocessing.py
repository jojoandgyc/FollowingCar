from __future__ import annotations


def non_max_suppression(boxes, max_bbox_overlap: float, scores=None):
    np = _np()
    if len(boxes) == 0:
        return []
    if max_bbox_overlap >= 1.0:
        return list(range(len(boxes)))

    boxes = boxes.astype("float32")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    idxs = np.argsort(scores) if scores is not None else np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = int(idxs[last])
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        overlap = (w * h) / (area[idxs[:last]] + area[i] - w * h)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))
    return pick


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for DeepSORT preprocessing") from exc
    return np
