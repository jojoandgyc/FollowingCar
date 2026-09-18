"""Normalize OpenNI Depth once, at ingestion, for unmirrored external UVC RGB.

Registration and mirroring are independent. Readback is mandatory: never guess
a flip from a failed setter. Device readback still needs a left/centre/right
scene check against the external UVC camera before motor operation.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DepthOrientation:
    depth_mirrored: bool
    color_mirrored: bool
    software_flip_x: bool
    coordinate_space: str = "external_uvc_unmirrored"

    def metadata(self):
        return asdict(self)

    def normalize(self, depth):
        # Caller owns the input array. Do not change RGB, bboxes, or turn signs.
        return depth[:, ::-1].copy() if self.software_flip_x else depth


def read_orientation(depth_stream, color_stream, logger, *, stage):
    states = {}
    for name, stream in (("depth", depth_stream), ("color", color_stream)):
        try:
            value = stream.get_mirroring_enabled()
            if value is None or value not in (False, True):
                raise ValueError(f"invalid mirror readback: {value!r}")
            states[name] = bool(value)
        except Exception as exc:
            raise RuntimeError(f"Astra {name}镜像状态无法读回，禁止使用方向未知的Depth") from exc
    if states["color"]:
        raise RuntimeError("Astra配准Color镜像仍开启，无法确认外部UVC坐标，禁止启用Depth")
    plan = DepthOrientation(states["depth"], states["color"], states["depth"])
    logger.info(
        "Astra orientation: stage=%s depth_mirrored=%s color_mirrored=%s "
        "software_flip_x=%s coordinate_space=%s RGB_transform=none",
        stage, plan.depth_mirrored, plan.color_mirrored,
        plan.software_flip_x, plan.coordinate_space,
    )
    if plan.software_flip_x:
        logger.warning("Astra Depth镜像仍开启：仅在Depth入口水平翻转一次，缓存/测距/快照共用转换后的数据")
    return plan


def configure_orientation(depth_stream, color_stream, logger):
    for name, stream in (("depth", depth_stream), ("color", color_stream)):
        try:
            before = stream.get_mirroring_enabled()
        except Exception:
            before = "unknown"
        logger.info("Astra orientation before: stream=%s mirrored=%s requested=False", name, before)
        try:
            stream.set_mirroring_enabled(False)
        except Exception as exc:
            # A known, already unmirrored state is safe even if writes fail.
            logger.warning("Astra mirror set failed: stream=%s error=%s; checking readback", name, exc)
    return read_orientation(depth_stream, color_stream, logger, stage="configured")
