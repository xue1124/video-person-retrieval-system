"""从描边 RGBA 图提取多边形（与 tests/yolo_osnet_query_search_streamlit 中逻辑一致）。"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def polygon_area_abs(poly: List[List[float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(s) * 0.5


def polygons_from_freedraw_rgba(
    rgba: np.ndarray,
    stroke_rgb: Tuple[int, int, int],
    color_tol: float = 60.0,
    min_area_frac: float = 0.00025,
    approx_frac: float = 0.003,
    max_polys: int = 16,
) -> List[List[List[float]]]:
    if rgba is None or rgba.size == 0 or rgba.ndim < 2:
        return []
    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    rgb = rgba[:, :, :3].astype(np.float32)
    tgt = np.array(stroke_rgb, dtype=np.float32).reshape(1, 1, 3)
    dist = np.linalg.norm(rgb - tgt, axis=2)
    mask = (dist <= color_tol).astype(np.uint8) * 255
    if rgba.shape[2] >= 4:
        mask = cv2.bitwise_and(mask, (rgba[:, :, 3] > 12).astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.dilate(mask, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas_area = float(max(1, h * w))
    polys: List[List[List[float]]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area_frac * canvas_area:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(float(approx_frac) * peri, 2.0)
        apx = cv2.approxPolyDP(c, eps, closed=True)
        if len(apx) < 3:
            continue
        polys.append([[float(p[0][0]), float(p[0][1])] for p in apx])
    polys.sort(key=polygon_area_abs, reverse=True)
    return polys[:max_polys]


def stroke_polygon_to_natural(
    poly_canvas: List[List[float]],
    canvas_w: int,
    canvas_h: int,
    natural_w: int,
    natural_h: int,
) -> List[List[float]]:
    sx = float(natural_w) / max(1, int(canvas_w))
    sy = float(natural_h) / max(1, int(canvas_h))
    out: List[List[float]] = []
    for p in poly_canvas:
        out.append([round(float(p[0]) * sx, 2), round(float(p[1]) * sy, 2)])
    return out


# ---------- 检索结果：脚底点与房间多边形（与 yolo_osnet_query_search_streamlit 一致） ----------


def point_in_polygon(px: float, py: float, poly: List[List[float]]) -> bool:
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        intersect = (yi > py) != (yj > py) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _dist_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby + 1e-9
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx, qy = ax + abx * t, ay + aby * t
    return float(math.hypot(px - qx, py - qy))


def min_dist_point_to_polygon(px: float, py: float, poly: List[List[float]]) -> float:
    if len(poly) < 2:
        return float("inf")
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        best = min(best, _dist_point_to_segment(px, py, ax, ay, bx, by))
    return best


def classify_point_to_room(px: float, py: float, rooms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rooms: [{"name": str, "polygon": [[x,y],...]}, ...]
    返回 name, method inside|nearest|none, dist_px
    """
    if not rooms:
        return {"name": "", "method": "none", "dist_px": 0.0}
    for r in rooms:
        name = str(r.get("name", "")).strip() or "未命名"
        poly = r.get("polygon") or []
        if len(poly) >= 3 and point_in_polygon(px, py, poly):
            return {"name": name, "method": "inside", "dist_px": 0.0}
    best_name = ""
    best_d = float("inf")
    for r in rooms:
        name = str(r.get("name", "")).strip() or "未命名"
        poly = r.get("polygon") or []
        if len(poly) < 2:
            continue
        d = min_dist_point_to_polygon(px, py, poly)
        if d < best_d:
            best_d = d
            best_name = name
    return {
        "name": best_name,
        "method": "nearest",
        "dist_px": round(best_d, 2) if math.isfinite(best_d) else 0.0,
    }


def bbox_foot_xy(bbox: List[float] | Tuple[float, ...] | Any) -> Tuple[float, float]:
    """行人框底边中点（与 Streamlit _bbox_foot_xy 一致）。"""
    x1, y1, x2, y2 = (float(bbox[i]) for i in range(4))
    return (x1 + x2) * 0.5, y2
