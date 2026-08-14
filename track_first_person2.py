# -*- coding: utf-8 -*-
"""
首人跟踪模块：只跟“第一个看到的人”，人群中不换目标；适配车前摄像头、人易出画、背景一直变、30fps。

- 仅用标准库：math, typing
- 算法：IoU + 面积相似度 + 运动预测（中心距离），粘性保持
  - 车在动、背景在变：同一人框在画面里会整体漂移 → 用上一帧中心+速度预测本帧中心，用“与预测中心距离”辅助匹配
  - 人出画：连续 3 帧无匹配才释放目标（防抖：避免检测某一帧漏检就立刻判丢人）。
  - 换人：只有「连续 600 帧（约 20 秒）没有和当前目标关联」后才允许重新锁「第一个看到的人」；否则重新有人入画时优先同检测器 ID 或「离上次目标位置足够近」才锁，避免跟成其他人。
- 提供目标 ID（target_id）与「是否为本帧确认目标」（is_current_target_confirmed），便于主逻辑打印。
- 计算量小，适合板子
"""

from typing import List, Tuple, Optional
import math


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _iou(bbox_a: Tuple[float, float, float, float], bbox_b: Tuple[float, float, float, float]) -> float:
    """交并比 [0, 1]，0 表示无重叠。"""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _size_similarity(bbox_a: Tuple[float, float, float, float], bbox_b: Tuple[float, float, float, float]) -> float:
    """面积相似度 [0, 1]。"""
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def _center_distance(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """两点欧氏距离（像素）。"""
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

def _bbox_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """宽高比 w/h（用于丢失后重入画的简单外观约束）。"""
    x1, y1, x2, y2 = bbox
    w = max(1e-6, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    return w / h


class FirstPersonTracker:
    """
    首人跟踪器：锁定第一个看到的人；适配车前摄像头、背景变、人易出画、30fps。
    - 运动预测：用上一帧中心+速度预测本帧中心，背景在变时用“与预测中心距离”辅助匹配（IoU 易掉）。
    - 人出画：连续 3 帧无匹配才释放（防抖）；换人：600 帧（约 20s）无目标后才允许锁「第一个看到的人」。
    persons 格式：[(bbox, track_id, conf, area), ...]，bbox = (x1,y1,x2,y2)；track_id 为检测器 unique_id。
    """

    def __init__(
        self,
        min_iou_threshold: float = 0.08,
        max_lost_frames: int = 3,
        frames_allow_new_id: int = 600,
        max_center_dist_ratio: float = 0.25,
        min_size_sim_for_center_match: float = 0.45,
        velocity_smooth: float = 0.4,
        max_reacquire_dist_ratio: float = 0.65,
        # 丢失后“重识别/找回”参数（仅用 bbox 几何，不依赖图像特征）
        min_reacquire_size_sim: float = 0.35,
        max_reacquire_aspect_diff: float = 0.90,
        reacquire_consecutive_frames: int = 2,
    ):
        """
        min_iou_threshold: IoU 高于此才可仅靠 IoU 匹配（略低一点，避免背景动一点就丢）
        max_lost_frames: 连续多少帧无匹配才释放目标（3 帧防抖，避免单帧漏检就判丢人）
        frames_allow_new_id: 释放后连续多少帧没有目标，才允许锁「第一个看到的人」即换人（600≈20s@30fps）
        max_center_dist_ratio / min_size_sim_for_center_match: 中心距离/面积相似度匹配阈值
        velocity_smooth: 速度平滑系数 (0~1)，预测更稳
        max_reacquire_dist_ratio: 未到换人帧数时，重新锁人仅当「离上次释放位置」距离 <= 该比例*ref_size，否则不锁（继续搜），避免跟成其他人。
        """
        self.min_iou_threshold = min_iou_threshold
        self.max_lost_frames = max_lost_frames
        self.frames_allow_new_id = frames_allow_new_id
        self.max_center_dist_ratio = max_center_dist_ratio
        self.min_size_sim_for_center_match = min_size_sim_for_center_match
        self.velocity_smooth = velocity_smooth
        self.max_reacquire_dist_ratio = max_reacquire_dist_ratio
        self.min_reacquire_size_sim = min_reacquire_size_sim
        self.max_reacquire_aspect_diff = max_reacquire_aspect_diff
        self.reacquire_consecutive_frames = max(1, int(reacquire_consecutive_frames))

        self._target_bbox: Optional[Tuple[float, float, float, float]] = None
        self._target_cx: float = 0.0
        self._target_cy: float = 0.0
        self._velocity_x: float = 0.0
        self._velocity_y: float = 0.0
        self._target_lost_frames: int = 0
        self._has_target: bool = False
        # 释放目标时记下当时中心，重新锁定时（且未到允许换人帧数）仅当有人离此足够近或同检测器 ID 才锁
        self._last_release_cx: Optional[float] = None
        self._last_release_cy: Optional[float] = None
        self._last_release_bbox: Optional[Tuple[float, float, float, float]] = None
        self._pending_reacquire_detector_id: Optional[int] = None
        self._reacquire_streak: int = 0
        # 已有多久没有目标（用于：超过 frames_allow_new_id 才允许跟「第一个看到的人」即换人）
        self._frames_without_target: int = 0
        # 当前跟随的逻辑目标编号（0,1,2...），允许换人时锁定新人才自增
        self._target_id: int = 0
        # 当前目标的检测器 unique_id，用于重入画时优先认同一人
        self._target_detector_id: Optional[int] = None
        # 上一帧输出是否为「跟踪匹配到的同一目标」（True）还是「刚重新锁定」（False）或未输出（False）
        self._last_output_confirmed: bool = False

    def update(
        self,
        persons: List[Tuple[Tuple[float, float, float, float], int, float, float]],
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> List[Tuple[Tuple[float, float, float, float], int, float, float]]:
        """
        每帧调用。传入当前帧所有检测到的人，可选传入 img_width/height 用于按比例算“中心距离”阈值。
        返回 0 或 1 个人。
        """
        if not self._has_target:
            self._frames_without_target += 1

        if not persons:
            self._last_output_confirmed = False
            if self._has_target:
                self._target_lost_frames += 1
                if self._target_lost_frames >= self.max_lost_frames:
                    self._last_release_cx = self._target_cx
                    self._last_release_cy = self._target_cy
                    self._last_release_bbox = self._target_bbox
                    self._has_target = False
                    self._target_bbox = None
                    self._velocity_x = 0.0
                    self._velocity_y = 0.0
                    self._pending_reacquire_detector_id = None
                    self._reacquire_streak = 0
            return []

        ref_size = 320.0
        if img_width is not None and img_height is not None:
            ref_size = min(img_width, img_height)
        max_center_dist = ref_size * self.max_center_dist_ratio

        # 尚未锁定目标：超过 frames_allow_new_id 才允许跟「第一个看到的人」即换人；否则仅当同检测器 ID 或离上次位置足够近才锁，避免跟成其他人
        if not self._has_target:
            allow_new_id = self._frames_without_target >= self.frames_allow_new_id
            if allow_new_id:
                best = max(persons, key=lambda x: x[3])
                self._last_release_cx = None
                self._last_release_cy = None
                self._last_release_bbox = None
                self._pending_reacquire_detector_id = None
                self._reacquire_streak = 0
                self._target_id += 1
                self._target_detector_id = best[1]
                self._target_bbox = best[0]
                self._target_cx, self._target_cy = _bbox_center(best[0])
                self._velocity_x = 0.0
                self._velocity_y = 0.0
                self._has_target = True
                self._target_lost_frames = 0
                self._frames_without_target = 0
                self._last_output_confirmed = False
                return [best]
            if self._last_release_cx is not None and self._last_release_cy is not None:
                max_reacquire_dist = ref_size * self.max_reacquire_dist_ratio
                # 目标刚释放不久：用 bbox 几何做“找回”(reacquire)，避免直接锁到最近的路人
                # 评分要点：
                # - 离上次释放中心越近越好
                # - 框面积越接近越好（大小比例）
                # - 宽高比越接近越好（粗粒度外观约束）
                rel_bbox = self._last_release_bbox
                rel_ar = _bbox_aspect_ratio(rel_bbox) if rel_bbox is not None else None
                best_score = -1e9
                best = None
                for p in persons:
                    bb = p[0]
                    cx, cy = _bbox_center(bb)
                    dist = _center_distance(self._last_release_cx, self._last_release_cy, cx, cy)
                    if dist > max_reacquire_dist:
                        continue
                    size_sim = _size_similarity(rel_bbox, bb) if rel_bbox is not None else 0.0
                    if size_sim < float(self.min_reacquire_size_sim):
                        continue
                    ar = _bbox_aspect_ratio(bb)
                    aspect_diff = abs(ar - rel_ar) / max(1e-6, rel_ar) if rel_ar is not None else 0.0
                    if rel_ar is not None and aspect_diff > float(self.max_reacquire_aspect_diff):
                        continue
                    # 距离分数：越近越高；size_sim [0,1]；aspect_diff 越小越好
                    dist_score = 1.0 / (1.0 + dist / max(1.0, ref_size * 0.20))
                    aspect_score = 1.0 / (1.0 + aspect_diff)
                    # 同 detector_id 加一点偏置（若 unique_id 稳定，会更容易找回）
                    id_bonus = 0.08 if (self._target_detector_id is not None and p[1] == self._target_detector_id) else 0.0
                    score = 0.50 * dist_score + 0.35 * size_sim + 0.15 * aspect_score + id_bonus
                    if score > best_score:
                        best_score = score
                        best = p

                if best is None:
                    self._last_output_confirmed = False
                    self._pending_reacquire_detector_id = None
                    self._reacquire_streak = 0
                    return []  # 没有候选满足阈值：继续等待，避免误锁路人

                # 连续多帧确认同一 detector_id 才真正锁定（防抖）
                cand_id = best[1]
                if self._pending_reacquire_detector_id != cand_id:
                    self._pending_reacquire_detector_id = cand_id
                    self._reacquire_streak = 1
                    self._last_output_confirmed = False
                    return []
                self._reacquire_streak += 1
                if self._reacquire_streak < self.reacquire_consecutive_frames:
                    self._last_output_confirmed = False
                    return []

                # 确认找回：锁定目标
                self._last_release_cx = None
                self._last_release_cy = None
                self._last_release_bbox = None
                self._pending_reacquire_detector_id = None
                self._reacquire_streak = 0
                self._target_detector_id = best[1]
                self._target_bbox = best[0]
                self._target_cx, self._target_cy = _bbox_center(best[0])
                self._velocity_x = 0.0
                self._velocity_y = 0.0
                self._has_target = True
                self._target_lost_frames = 0
                self._frames_without_target = 0
                self._last_output_confirmed = False
                return [best]
            # 无上次释放位置（如首次启动）：选面积最大
            best = max(persons, key=lambda x: x[3])
            self._target_detector_id = best[1]
            self._target_bbox = best[0]
            self._target_cx, self._target_cy = _bbox_center(best[0])
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            self._has_target = True
            self._target_lost_frames = 0
            self._frames_without_target = 0
            self._last_output_confirmed = False
            return [best]

        # 预测本帧目标中心（车在动、背景在变时框会整体漂移）
        pred_cx = self._target_cx + self._velocity_x
        pred_cy = self._target_cy + self._velocity_y

        best_score = -1.0
        best_person = None
        for one in persons:
            bbox = one[0]
            cx, cy = _bbox_center(bbox)
            iou = _iou(self._target_bbox, bbox)
            size_sim = _size_similarity(self._target_bbox, bbox)
            dist = _center_distance(pred_cx, pred_cy, cx, cy)

            # 匹配条件：(1) IoU 足够 或 (2) 中心离预测很近且面积相似，避免背景一变就丢
            iou_ok = iou >= self.min_iou_threshold
            center_ok = dist <= max_center_dist and size_sim >= self.min_size_sim_for_center_match
            if not (iou_ok or center_ok):
                continue

            # 得分：IoU 为主，面积相似度 + 中心距离（越近分越高）
            center_score = 1.0 / (1.0 + dist / (ref_size * 0.15)) if ref_size > 0 else 0.0
            score = 0.55 * iou + 0.25 * size_sim + 0.2 * center_score
            if score > best_score:
                best_score = score
                best_person = one

        if best_person is not None:
            bbox = best_person[0]
            cx, cy = _bbox_center(bbox)
            self._target_detector_id = best_person[1]
            self._last_output_confirmed = True
            # 更新速度（平滑），用于下一帧预测
            new_vx = cx - self._target_cx
            new_vy = cy - self._target_cy
            self._velocity_x = self.velocity_smooth * new_vx + (1.0 - self.velocity_smooth) * self._velocity_x
            self._velocity_y = self.velocity_smooth * new_vy + (1.0 - self.velocity_smooth) * self._velocity_y
            self._target_bbox = bbox
            self._target_cx = cx
            self._target_cy = cy
            self._target_lost_frames = 0
            return [best_person]

        self._last_output_confirmed = False
        self._target_lost_frames += 1
        if self._target_lost_frames >= self.max_lost_frames:
            self._last_release_cx = self._target_cx
            self._last_release_cy = self._target_cy
            self._last_release_bbox = self._target_bbox
            self._has_target = False
            self._target_bbox = None
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            self._pending_reacquire_detector_id = None
            self._reacquire_streak = 0
        return []

    def get_target_id(self) -> int:
        """当前跟随的逻辑目标编号（0, 1, 2...），换人时自增。"""
        return self._target_id

    def is_current_target_confirmed(self) -> bool:
        """本帧输出的人是否为「跟踪匹配到的同一目标」；False 表示刚重新锁定或本帧未输出人。"""
        return self._last_output_confirmed
