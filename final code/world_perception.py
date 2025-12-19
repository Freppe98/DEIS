#!/usr/bin/env python3
"""
world_perception.py

Thin wrapper around the Mermaid APIs:
- RobotPositionAPI (ROS2) -> robot row/col (+ optional heading from 2 spirals)
- HandRecognitionAPI -> gesture + pixel position
- GPSOverlay -> pixel->grid cell (with optional height correction)
- layout-api -> load grid + find home cell

- "poll latest snapshot" to use
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal, List, Dict
import os
import importlib.util
import sys
import math
import time


GestureType = Literal["NONE", "OPEN_PALM", "CLOSED_FIST"]


@dataclass
class PerceptionSnapshot:
    timestamp: float

    # Robot
    robot_visible: bool
    robot_cell: Optional[Tuple[int, int]]          # (row, col)
    robot_server_px: Optional[Tuple[float, float]] # (x, y) in GPS server coords
    robot_heading_deg: Optional[float]             # 0=+x (east), +90=+y (south) in server frame
    robot_heading_card: Optional[str]              # 'N','E','S','W'

    # Gesture
    gesture_visible: bool
    gesture_type: GestureType
    gesture_cell: Optional[Tuple[int, int]]        # (row, col)
    gesture_server_px: Optional[Tuple[float, float]]

    # Static
    home_cell: Optional[Tuple[int, int]]


# -----------------------------------------------------------------------------
# Dynamic loader (because API files are named with hyphens)
# -----------------------------------------------------------------------------

def _load_module(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


# -----------------------------------------------------------------------------
# Globals (simple)
# -----------------------------------------------------------------------------

_ros_api = None
_hand_api = None
_overlay = None
_grid: Optional[List[List[int]]] = None
_home_cell: Optional[Tuple[int, int]] = None

_cfg: Dict = {
    "robot_pos_id": 0,       # spiral ID for robot "position" tag
    "robot_head_id": 1,      # spiral ID for robot "heading" tag (rear); heading vector: head->pos
    "hand_height_mm": 1000.0,
    "min_certainty": 0.25,
    "max_speed": 500.0,
    "ros_topic": "robotPositions",
    "ros_msg_type": "string",  # 'string' or 'float32multiarray'
    "hand_stream_url": None,
    "hand_show_window": False,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _wrap_angle_deg_0_360(a: float) -> float:
    return (a % 360.0 + 360.0) % 360.0


def _cardinal_from_deg(angle_deg: float) -> str:
    """
    convention:
    - 0 deg ~ East, 90 deg ~ South, 180 ~ West, 270 ~ North
    """
    a = _wrap_angle_deg_0_360(angle_deg)
    if a >= 315 or a < 45:
        return "E"
    if 45 <= a < 135:
        return "S"
    if 135 <= a < 225:
        return "W"
    return "N"


def _find_home_cell(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
    # layout-api defines HOME=2
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if int(v) == 2:
                return (r, c)
    return None


def _infer_cell_from_spiral_row(row_val: float, col_val: float) -> Optional[Tuple[int, int]]:
    """
    RobotPositionAPI returns SpiralRow(row, col, angle, id, certainty).
    Those row/col can be either:
      - grid indices (0..rows-1, 0..cols-1)
      - GPS server pixels (y, x) i.e. row ~ y, col ~ x

    We detect by comparing to grid size.
    """
    global _grid, _overlay
    if _grid is None or _overlay is None:
        return None

    rows = len(_grid)
    cols = len(_grid[0]) if rows else 0
    rr = float(row_val)
    cc = float(col_val)

    # Looks like cell indices?
    if 0 <= rr < rows and 0 <= cc < cols:
        return (int(round(rr)), int(round(cc)))

    # Otherwise treat as server pixels: (x=col, y=row)
    cell = _overlay.get_grid_cell(cc, rr)
    if cell and cell.get("in_bounds", False):
        return (int(cell["row"]), int(cell["col"]))
    return None


def _server_px_from_spiral(row_val: float, col_val: float) -> Tuple[float, float]:
    # SpiralRow uses (row, col) but in pixel mode row~y, col~x
    return (float(col_val), float(row_val))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def start_perception(
    overlay_json_path: str,
    grid_json_path: str,
    *,
    ros_topic: str = "robotPositions",
    ros_msg_type: str = "string",
    robot_pos_id: int = 0,
    robot_head_id: int = 1,
    min_certainty: float = 0.25,
    max_speed: float = 500.0,
    hand_stream_url: Optional[str] = None,
    hand_show_window: bool = False,
    hand_height_mm: float = 1000.0,
):
    """
    One-time init. Starts ROS2 + hand recognition and loads overlay + grid.

    NOTE: Call stop_perception() on exit.
    """
    global _ros_api, _hand_api, _overlay, _grid, _home_cell, _cfg

    base = Path(__file__).resolve().parent

    ros_mod = _load_module(base / "Mermaid-Integration/apis/ros2-api/ros2-api.py", "mi_ros2_api_min")
    hand_mod = _load_module(base / "Mermaid-Integration/apis/hand-recognition-api/interaction-api.py", "mi_hand_api_min")
    ovl_mod = _load_module(base / "Mermaid-Integration/apis/overlay-api/overlay-api.py", "mi_overlay_api_min")
    lay_mod = _load_module(base / "Mermaid-Integration/apis/layout-api/layout-api.py", "mi_layout_api_min")

    RobotPositionAPI = getattr(ros_mod, "RobotPositionAPI")
    HandRecognitionAPI = getattr(hand_mod, "HandRecognitionAPI")
    GPSOverlay = getattr(ovl_mod, "GPSOverlay")
    load_grid = getattr(lay_mod, "_load_grid_internal")

    _cfg.update({
        "robot_pos_id": int(robot_pos_id),
        "robot_head_id": int(robot_head_id),
        "hand_height_mm": float(hand_height_mm),
        "min_certainty": float(min_certainty),
        "max_speed": float(max_speed),
        "ros_topic": str(ros_topic),
        "ros_msg_type": str(ros_msg_type),
        "hand_stream_url": hand_stream_url,
        "hand_show_window": bool(hand_show_window),
    })

    _overlay = GPSOverlay(overlay_json_path)
    _grid = load_grid(grid_json_path)
    _home_cell = _find_home_cell(_grid) if _grid else None

    _ros_api = RobotPositionAPI(
        topic=_cfg["ros_topic"],
        msg_type=_cfg["ros_msg_type"],
        min_certainty=_cfg["min_certainty"],
        max_speed=_cfg["max_speed"],
    )
    _ros_api.start()

    _hand_api = HandRecognitionAPI(stream_url=hand_stream_url, show_window=hand_show_window)
    _hand_api.start()


def stop_perception():
    """Stop ROS2 + hand threads."""
    global _ros_api, _hand_api
    try:
        if _hand_api is not None:
            _hand_api.stop()
    finally:
        _hand_api = None
    try:
        if _ros_api is not None:
            _ros_api.stop()
    finally:
        _ros_api = None


def get_grid() -> List[List[int]]:
    """Return the loaded integer grid (FREE=0, OBSTACLE=1, HOME=2)."""
    return _grid if _grid is not None else []


def get_home_cell() -> Optional[Tuple[int, int]]:
    """Return cached HOME cell from grid (row,col)."""
    return _home_cell


def get_snapshot() -> PerceptionSnapshot:
    """
    Poll the latest robot + hand state.
    This is what new_main.py call repeatedly.
    """
    global _ros_api, _hand_api, _overlay, _grid, _home_cell, _cfg

    now = time.time()

    # ---- Robot ----
    robot_visible = False
    robot_cell = None
    robot_server_px = None
    heading_deg = None
    heading_card = None

    if _ros_api is not None:
        pos_id = int(_cfg["robot_pos_id"])
        head_id = int(_cfg["robot_head_id"])

        r_pos = _ros_api.getPosition(pos_id)
        r_head = _ros_api.getPosition(head_id)

        if r_pos is not None:
            robot_visible = True
            robot_cell = _infer_cell_from_spiral_row(r_pos.row, r_pos.col)
            robot_server_px = _server_px_from_spiral(r_pos.row, r_pos.col)

        # Heading from head -> pos (in server frame)
        if r_pos is not None and r_head is not None:
            x0, y0 = _server_px_from_spiral(r_pos.row, r_pos.col)
            x1, y1 = _server_px_from_spiral(r_head.row, r_head.col)
            dx = x0 - x1
            dy = y0 - y1
            heading_deg = math.degrees(math.atan2(dy, dx))
            heading_deg = _wrap_angle_deg_0_360(heading_deg)
            heading_card = _cardinal_from_deg(heading_deg)

    # ---- Gesture ----
    gesture_visible = False
    gesture_type: GestureType = "NONE"
    gesture_cell = None
    gesture_server_px = None

    if _hand_api is not None and _overlay is not None:
        hs = _hand_api.get_state()
        if hs is not None:
            x0, y0 = float(hs.position[0]), float(hs.position[1])
            # swap for coordinate systems that report (y,x) instead of (x,y)
            #x, y = y0, x0
            x, y = x0, y0
            gesture_server_px = (x, y)
            gesture_visible = True

            if hs.gesture == "PALM":
                gesture_type = "OPEN_PALM"
            elif hs.gesture == "FIST":
                gesture_type = "CLOSED_FIST"
            else:
                gesture_type = "NONE"

            # Height-corrected mapping for hand
            #cell = _overlay.get_grid_cell_with_height_offset(
            #    x, y, height_mm=float(_cfg["hand_height_mm"])
            #)
            cell = _overlay.get_grid_cell(x, y)

            #print(f"Hand at server px=({x:.1f},{y:.1f}), cell={cell}, cellB={cellB}")
            #print((x,y), "grid:", (cell["row"], cell["col"]), (cellB["row"], cellB["col"]))

            if cell and cell.get("in_bounds", False):
                gesture_cell = (int(cell["row"]), int(cell["col"]))

    return PerceptionSnapshot(
        timestamp=now,
        robot_visible=robot_visible,
        robot_cell=robot_cell,
        robot_server_px=robot_server_px,
        robot_heading_deg=heading_deg,
        robot_heading_card=heading_card,
        gesture_visible=gesture_visible,
        gesture_type=gesture_type,
        gesture_cell=gesture_cell,
        gesture_server_px=gesture_server_px,
        home_cell=_home_cell,
    )


# -----------------------------------------------------------------------------
# Grid cell neighborhood helpers (exported for controller to use)
# -----------------------------------------------------------------------------

def rc_neighbors8(rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return 8-connected neighbor cells within grid bounds."""
    global _grid
    if rc is None:
        return []
    r, c = rc
    nbrs: List[Tuple[int, int]] = []
    if _grid is None or not _grid:
        # No bounds info; still return theoretical neighbors
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nbrs.append((r + dr, c + dc))
        return nbrs

    rows = len(_grid)
    cols = len(_grid[0]) if rows else 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                nbrs.append((rr, cc))
    return nbrs


def rc_is_neighbor_or_same(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Return True if cells are identical or 8-neighbors (Chebyshev distance <= 1)."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    return b in rc_neighbors8(a)


def rc_neighbors4(rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return 4-connected neighbor cells within grid bounds (N,E,S,W)."""
    global _grid
    if rc is None:
        return []
    r, c = rc
    candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    if _grid is None or not _grid:
        return candidates
    rows = len(_grid)
    cols = len(_grid[0]) if rows else 0
    return [(rr, cc) for rr, cc in candidates if 0 <= rr < rows and 0 <= cc < cols]


def rc_is_4neighbor_or_same(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """True if same or 4-neighbor (Manhattan distance == 1)."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    return b in rc_neighbors4(a)


# -----------------------------------------------------------------------------
# Cell distance utilities
# -----------------------------------------------------------------------------

def _cell_center_rect(a: Tuple[int, int]) -> Tuple[float, float]:
    """
    Return rectified canvas coordinates (pixels) of the center of a grid cell.
    Uses overlay's arena bounds and grid dimensions.
    """
    global _overlay
    if a is None:
        raise ValueError("Cell cannot be None")
    if _overlay is None:
        raise RuntimeError("Overlay not initialized. Call start_perception() first.")
    row, col = int(a[0]), int(a[1])
    left = float(_overlay.arena_bounds["left"])  # type: ignore
    top = float(_overlay.arena_bounds["top"])    # type: ignore
    right = float(_overlay.arena_bounds["right"])  # type: ignore
    bottom = float(_overlay.arena_bounds["bottom"])  # type: ignore
    cols = int(_overlay.grid_cols)  # type: ignore
    rows = int(_overlay.grid_rows)  # type: ignore
    cell_width = (right - left) / max(cols, 1)
    cell_height = (bottom - top) / max(rows, 1)
    center_x = left + (col + 0.5) * cell_width
    center_y = top + (row + 0.5) * cell_height
    return center_x, center_y


def rc_distance_px(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Euclidean distance between two grid cells (row,col) measured in rectified pixels.
    """
    x0, y0 = _cell_center_rect(a)
    x1, y1 = _cell_center_rect(b)
    return math.hypot(x1 - x0, y1 - y0)


def rc_distance_mm(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Euclidean distance between two grid cells (row,col) measured in millimeters.
    Requires real-world calibration in overlay (mm_per_pixel_x/y); raises ValueError if unavailable.
    """
    global _overlay
    if _overlay is None:
        raise RuntimeError("Overlay not initialized. Call start_perception() first.")
    if not getattr(_overlay, "real_world_available", False):
        raise ValueError("Real-world calibration not available in overlay.")
    mm_x = getattr(_overlay, "mm_per_pixel_x", None)
    mm_y = getattr(_overlay, "mm_per_pixel_y", None)
    if mm_x in (None, 0) or mm_y in (None, 0):
        raise ValueError("mm_per_pixel_x/y missing or zero in overlay calibration.")
    x0, y0 = _cell_center_rect(a)
    x1, y1 = _cell_center_rect(b)
    dx_mm = (x1 - x0) * float(mm_x)
    dy_mm = (y1 - y0) * float(mm_y)
    return math.hypot(dx_mm, dy_mm)

