# Comments in English only
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ------------ Types ------------
# candidate: (confidence, (x1,y1,x2,y2)) in pixel coordinates
Candidate = Tuple[float, Tuple[float, float, float, float]]


# ------------ Config ------------
@dataclass
class TargetLockConfig:
    lock_iou_min: float = 0.25     # minimum IoU to keep lock
    lock_max_miss_fr: int = 10     # consecutive frames allowed to miss before reset
    prefer_roi: bool = True        # prefer candidates whose center is inside ROI


# ------------ Geometry ------------
def iou(ax1: float, ay1: float, ax2: float, ay2: float,
        bx1: float, by1: float, bx2: float, by2: float) -> float:
    """Intersection-over-Union between two axis-aligned boxes."""
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a + b - inter
    return (inter / denom) if denom > 0.0 else 0.0


# ------------ Target Lock ------------
class TargetLock:
    """
    Tracks a single target box across frames using IoU association.
    Use:
        lock = TargetLock(TargetLockConfig(...))
        lock.acquire(candidates, roi_rect=(rx1, ry1, rx2, ry2))
        lock.update(candidates)
        lock.box -> current (x1,y1,x2,y2) or None
        lock.active -> bool
    """

    def __init__(self, config: Optional[TargetLockConfig] = None):
        self.cfg = config or TargetLockConfig()
        self.box: Optional[Tuple[float, float, float, float]] = None
        self.miss: int = 0
        self.active: bool = False

    def reset(self) -> None:
        self.box = None
        self.miss = 0
        self.active = False

    def acquire(self, candidates: List[Candidate],
                roi_rect: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Pick best candidate by confidence (optionally preferring inside-ROI)."""
        if not candidates:
            return False

        pool = candidates
        if roi_rect and self.cfg.prefer_roi:
            rx1, ry1, rx2, ry2 = roi_rect
            in_roi: List[Candidate] = []
            out_roi: List[Candidate] = []
            for conf, (x1, y1, x2, y2) in candidates:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                (in_roi if (rx1 <= cx <= rx2 and ry1 <= cy <= ry2) else out_roi).append((conf, (x1, y1, x2, y2)))
            pool = in_roi if in_roi else out_roi

        best = max(pool, key=lambda t: t[0])
        self.box = best[1]
        self.miss = 0
        self.active = True
        return True

    def update(self, candidates: List[Candidate]) -> bool:
        """Associate by highest IoU with last box; enforce miss budget."""
        if not self.active or self.box is None:
            return False

        if not candidates:
            self.miss += 1
            if self.miss > self.cfg.lock_max_miss_fr:
                self.reset()
            return self.active

        bx1, by1, bx2, by2 = self.box
        best_iou, best_box = -1.0, None
        for _, (x1, y1, x2, y2) in candidates:
            ov = iou(bx1, by1, bx2, by2, x1, y1, x2, y2)
            if ov > best_iou:
                best_iou, best_box = ov, (x1, y1, x2, y2)

        if best_iou >= self.cfg.lock_iou_min:
            self.box = best_box
            self.miss = 0
        else:
            self.miss += 1
            if self.miss > self.cfg.lock_max_miss_fr:
                self.reset()

        return self.active


def find_person_candidates(
    yolo_result: object,
    class_names: dict,
    min_conf: float,
    min_box_frac_h: float,
    frame_height: int
) -> List[Candidate]:
    """
    (TESTING) Processes raw YOLO results to find any 'person' as a candidate.
    """
    candidates: List[Candidate] = []
    if not hasattr(yolo_result, 'boxes') or yolo_result.boxes is None:
        return []

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    clss = yolo_result.boxes.cls.cpu().numpy().astype(int)
    confs = yolo_result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cid, p in zip(boxes, clss, confs):
        if p < min_conf:
            continue
        
        box_h = y2 - y1
        if box_h < (min_box_frac_h * frame_height):
            continue

        class_name = class_names.get(int(cid), "")
        
        if class_name == 'person':
            coords = (float(x1), float(y1), float(x2), float(y2))
            candidates.append((float(p), coords))

    return candidates


def find_person_on_bench_candidates(
    yolo_result: object,
    class_names: dict,
    min_conf: float,
    min_box_frac_h: float,
    frame_height: int
) -> List[Candidate]:
    """
    (PRODUCTION) Processes raw YOLO results to find "person on bench" scenes.
    """
    persons = []
    benches = []
    if not hasattr(yolo_result, 'boxes') or yolo_result.boxes is None:
        return []

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    clss = yolo_result.boxes.cls.cpu().numpy().astype(int)
    confs = yolo_result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cid, p in zip(boxes, clss, confs):
        if p < min_conf:
            continue
        
        box_h = y2 - y1
        if box_h < (min_box_frac_h * frame_height):
            continue

        class_name = class_names.get(int(cid), "")
        coords = (float(x1), float(y1), float(x2), float(y2))
        if class_name == 'person':
            persons.append({'conf': float(p), 'box': coords})
        elif class_name == 'bench':
            benches.append({'conf': float(p), 'box': coords})

    candidates: List[Candidate] = []
    for p_obj in persons:
        px1, py1, px2, py2 = p_obj['box']
        person_center_x = (px1 + px2) / 2

        for b_obj in benches:
            bx1, by1, bx2, by2 = b_obj['box']

            is_horizontally_aligned = (bx1 < person_center_x < bx2)
            is_vertically_aligned = (by1 < py2 < by2)

            if is_horizontally_aligned and is_vertically_aligned:
                union_box = (min(px1, bx1), min(py1, by1), max(px2, bx2), max(py2, by2))
                candidates.append((p_obj['conf'], union_box))
                break

    return candidates
