"""Recorder-only numeric diagnostics for actual depth sampling regions.

Values are per-region raw cluster medians, NOT the fused control distance.
No colour map, depth resampling, new sensor read or control output.
"""
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class DepthVideoSample:
    # Live cache keeps None and scalar metadata, not whole depth arrays.
    depth: object
    metadata: dict


def region_distance(region, metadata):
    """Prefer the exact selected cluster; otherwise strongest candidate."""
    chosen = metadata.get("selected_region_distances", {}).get(region["name"])
    if chosen is not None and math.isfinite(float(chosen)) and float(chosen) > 0:
        return float(chosen), "selected"
    candidates = [c for c in region.get("clusters", ())
                  if c.get("distance_m") is not None and math.isfinite(float(c["distance_m"]))
                  and float(c["distance_m"]) > 0]
    if not candidates:
        return None, "unavailable"
    return float(max(candidates, key=lambda c: c.get("pixels", 0))["distance_m"]), "candidate"


@dataclass(frozen=True)
class DepthVideoView:
    status: str = "no_sample"
    sample: object = None
    skew_ms: object = None

    @classmethod
    def from_sample(cls, sample, capture_id, capture_timestamp, target_id=None):
        if sample is None:
            return cls()
        meta = sample.metadata
        if target_id is not None and meta.get("target_id") != target_id:
            return cls("target_mismatch")
        try:
            skew = 1000 * (float(meta["sample_timestamp"]) - capture_timestamp)
            if not math.isfinite(skew):
                return cls("invalid_time")
        except (KeyError, TypeError, ValueError):
            return cls("invalid_time")
        if abs(skew) > 80 + 1e-6:
            return cls("unaligned", None, skew)
        return cls("sampled", sample, skew)

    def csv_values(self):
        meta = {} if self.sample is None else self.sample.metadata
        regions = [dict(name=r["name"], distance_m=region_distance(r, meta)[0],
                        kind=region_distance(r, meta)[1], valid=r.get("valid"))
                   for r in meta.get("regions", ())[:5]]
        return (self.status, "" if self.skew_ms is None else f"{self.skew_ms:.1f}",
                meta.get("sample_timestamp", ""), meta.get("target_id", ""),
                meta.get("detail", ""), meta.get("evidence_capture_frame_id", ""),
                json.dumps(regions, separators=(",", ":"), allow_nan=False))


def draw_depth_overlay(image, view, cv2):
    """Five small labelled boxes, only on the recorder's private RGB."""
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_scale = max(.3, min(.5, width/1600))

    def text(value, at, scale=panel_scale, color=(255,255,255)):
        cv2.putText(image, value, at, font, scale, (0,0,0), 3)
        cv2.putText(image, value, at, font, scale, color, 1)

    label = "DEPTH ROI " + view.status.upper()
    if view.skew_ms is not None:
        label += f" dt={view.skew_ms:+.0f}ms"
    x, y = max(8, width-270), 55
    if view.sample is not None:
        meta = view.sample.metadata
        size = meta.get("depth_size")
        if size is None and view.sample.depth is not None:
            size = (view.sample.depth.shape[1], view.sample.depth.shape[0])
        if size is None:
            text("DEPTH ROI NO_GEOMETRY", (x,y))
            return
        dw, dh = size
        for region in meta.get("regions", ())[:5]:
            left, top, right, bottom = map(int, region["roi"])
            left, top, right, bottom = max(0,left), max(0,top), min(dw,right), min(dh,bottom)
            if right <= left or bottom <= top:
                continue
            x0, y0 = int(left*width/dw), int(top*height/dh)
            x1, y1 = min(width-1, math.ceil(right*width/dw)-1), min(height-1, math.ceil(bottom*height/dh)-1)
            distance, kind = region_distance(region, meta)
            # Neither selected nor candidate distance is permission to drive.
            value = "n/a" if distance is None else f"{distance:.2f}m"
            color = (255,255,0) if kind == "selected" else (180,180,180)
            cv2.rectangle(image, (x0,y0), (x1,y1), color, 1)
            scale = .40
            (tw,th), baseline = cv2.getTextSize(value, font, scale, 1)
            scale *= min(1., max(1,x1-x0-4)/max(1,tw), max(1,y1-y0-4)/max(1,th+baseline))
            (tw,th), baseline = cv2.getTextSize(value, font, scale, 1)
            tx, ty = x0+2, y0+th+2
            cv2.rectangle(image, (tx-1,ty-th-1), (min(x1,tx+tw+1),min(y1,ty+baseline+1)), (0,0,0), -1)
            cv2.putText(image, value, (tx,ty), font, scale, (255,255,255), 1)
            if y1-ty >= 14:
                mark = "S" if kind == "selected" else "C" if kind == "candidate" else "--"
                text(mark, (tx,min(y1-2,ty+12)), scale=.30, color=color)
        label += f" U{meta.get('target_id', '?')}"
        text(str(meta.get("rejection_reason") or meta.get("detail", ""))[:48], (x,y+15))
        text(f"ROI CAP {meta.get('evidence_capture_frame_id', '?')} S=selected C=candidate", (x,y+30))
    text(label, (x,y))
