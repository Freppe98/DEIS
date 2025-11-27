#!/usr/bin/env python3
"""
nav.py — Grid navigator that moves cell-by-cell using Mermaid-Integration APIs

Features
- Loads 2D grid from layout API (grid.json).
- Plans a shortest path (BFS) from start cell to goal cell through free cells.
- Uses RedBot ticks to move one grid at a time; turns 90° between segments.
- Initializes orientation from ROS2 heading with configurable mapping.

Environment overrides
- T3_TICKS_PER_CELL (default: 192)
- T3_TURN_TICKS (default: 96)
- T3_FWD_SPEED (default: 100)
- T3_TURN_SPEED (default: 70)
- T3_CMD_TIMEOUT (default: 8.0)
- T3_SPIRAL_IDS (default: "0,1")
- T3_HEADING_OFFSET_DEG (default: 0.0)
- T3_HEADING_ZERO_DEG_DIR (default: "E", options: E,N,W,S)
- T3_HEADING_SIGN (default: "ccw", options: ccw,cw)

Usage (example):
  from nav import GridNavigator
  nav = GridNavigator(ros_api, overlay)
  nav.goto((start_row, start_col), (goal_row, goal_col))
"""

from __future__ import annotations

import os
import math
from collections import deque
from typing import List, Tuple, Optional

import redbot

# Optional types
Cell = Tuple[int, int]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_list_int(name: str, default: str) -> List[int]:
    raw = os.environ.get(name, default)
    out: List[int] = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


class GridNavigator:
    def __init__(self, ros_api, overlay,
                 ticks_per_cell: int = None,
                 turn_ticks: int = None,
                 fwd_speed: int = None,
                 turn_speed: int = None,
                 cmd_timeout: float = None,
                 candidate_ids: Optional[List[int]] = None):
        self.ros_api = ros_api
        self.overlay = overlay
        self.ticks_per_cell = ticks_per_cell if ticks_per_cell is not None else _env_int('T3_TICKS_PER_CELL', 192)
        self.turn_ticks = turn_ticks if turn_ticks is not None else _env_int('T3_TURN_TICKS', 96)
        self.fwd_speed = fwd_speed if fwd_speed is not None else _env_int('T3_FWD_SPEED', 100)
        self.turn_speed = turn_speed if turn_speed is not None else _env_int('T3_TURN_SPEED', 70)
        self.cmd_timeout = cmd_timeout if cmd_timeout is not None else _env_float('T3_CMD_TIMEOUT', 8.0)
        self.candidate_ids = candidate_ids if candidate_ids is not None else _env_list_int('T3_SPIRAL_IDS', '0,1')

        # Orientation model
        self.heading_offset_deg = _env_float('T3_HEADING_OFFSET_DEG', 0.0)
        self.heading_zero_dir = os.environ.get('T3_HEADING_ZERO_DEG_DIR', 'E').upper()
        self.heading_sign = os.environ.get('T3_HEADING_SIGN', 'ccw').lower()  # 'ccw' or 'cw'
        self.current_dir = 'E'  # will be set during init from ROS2 angle

    # ---- Grid helpers ----
    @staticmethod
    def load_grid_as_ints(get_map_json_func) -> List[List[int]]:
        data = get_map_json_func()
        if not isinstance(data, list):
            return []
        grid: List[List[int]] = []
        for row in data:
            row_ints: List[int] = []
            for cell in row:
                if isinstance(cell, str):
                    cell = cell.upper()
                    if cell == 'O':
                        row_ints.append(0)
                    elif cell == 'X':
                        row_ints.append(1)
                    elif cell == 'H':
                        row_ints.append(0)  # HOME is traversable
                    else:
                        row_ints.append(0)
                else:
                    try:
                        row_ints.append(int(cell))
                    except Exception:
                        row_ints.append(0)
            grid.append(row_ints)
        return grid

    @staticmethod
    def bfs(grid: List[List[int]], start: Cell, goal: Cell) -> List[Cell]:
        if not grid:
            return []
        R, C = len(grid), len(grid[0])
        sr, sc = start
        gr, gc = goal
        if not (0 <= sr < R and 0 <= sc < C and 0 <= gr < R and 0 <= gc < C):
            return []
        if grid[sr][sc] == 1 or grid[gr][gc] == 1:
            return []
        dirs = [(-1,0),(0,1),(1,0),(0,-1)]  # N,E,S,W
        prev = [[None for _ in range(C)] for __ in range(R)]
        q = deque([start])
        prev[sr][sc] = start
        while q:
            r,c = q.popleft()
            if (r,c) == goal:
                break
            for dr,dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and prev[nr][nc] is None and grid[nr][nc] != 1:
                    prev[nr][nc] = (r,c)
                    q.append((nr,nc))
        if prev[gr][gc] is None:
            return []
        # reconstruct
        path: List[Cell] = []
        cur = (gr,gc)
        while cur != (sr,sc):
            path.append(cur)
            cur = prev[cur[0]][cur[1]]
        path.append((sr,sc))
        path.reverse()
        return path

    # ---- Orientation mapping ----
    def _angle_to_dir(self, angle_rad: float) -> str:
        if angle_rad is None:
            return self.heading_zero_dir
        deg = math.degrees(angle_rad) - self.heading_offset_deg
        # wrap to [0,360)
        deg = (deg % 360 + 360) % 360
        # map to quadrant
        # Define 0 deg direction label and sign
        order_ccw = ['E','N','W','S']
        order_cw  = ['E','S','W','N']
        base_order = order_ccw if self.heading_sign == 'ccw' else order_cw
        # rotate order so index 0 corresponds to configured zero direction
        zero_idx = base_order.index(self.heading_zero_dir) if self.heading_zero_dir in base_order else 0
        order = base_order[zero_idx:] + base_order[:zero_idx]
        # Determine nearest 90 deg multiple
        idx = int(((deg + 45) // 90) % 4)
        return order[idx]

    @staticmethod
    def dir_delta(d: str) -> Tuple[int,int]:
        if d == 'N': return (-1,0)
        if d == 'E': return (0,1)
        if d == 'S': return (1,0)
        if d == 'W': return (0,-1)
        return (0,1)

    @staticmethod
    def turn_steps(from_dir: str, to_dir: str) -> int:
        order = ['N','E','S','W']
        a = order.index(from_dir)
        b = order.index(to_dir)
        diff = (b - a) % 4
        # diff: 0=none,1=right,2=about,3=left
        return diff

    # ---- Motion primitives ----
    def _turn_right(self) -> bool:
        return redbot.send_ticks_blocking(self.turn_ticks, -self.turn_ticks,
                                          self.turn_speed, self.turn_speed,
                                          timeout=self.cmd_timeout)

    def _turn_left(self) -> bool:
        return redbot.send_ticks_blocking(-self.turn_ticks, self.turn_ticks,
                                          self.turn_speed, self.turn_speed,
                                          timeout=self.cmd_timeout)

    def _turn_about(self) -> bool:
        ok1 = self._turn_right()
        ok2 = self._turn_right()
        return bool(ok1 and ok2)

    def _forward_one_cell(self) -> bool:
        return redbot.send_ticks_blocking(self.ticks_per_cell, self.ticks_per_cell,
                                          self.fwd_speed, self.fwd_speed,
                                          timeout=self.cmd_timeout)

    # ---- ROS helpers ----
    def _read_pose(self):
        rows = self.ros_api.getPosition()
        if not rows:
            return None
        if self.candidate_ids:
            rows = [r for r in rows if getattr(r, 'id', None) in self.candidate_ids]
        if not rows:
            return None
        # choose highest certainty
        return max(rows, key=lambda r: getattr(r,'certainty',0.0))

    # ---- Public API ----
    def goto(self, start_cell: Cell, goal_cell: Cell, get_map_json_func) -> bool:
        # Initialize orientation from ROS heading if available
        pose = self._read_pose()
        if pose is not None:
            self.current_dir = self._angle_to_dir(getattr(pose, 'angle', None))
        # Load grid
        grid = GridNavigator.load_grid_as_ints(get_map_json_func)
        if not grid:
            print('[nav] ERROR: grid not loaded')
            return False
        # Plan path
        path = GridNavigator.bfs(grid, start_cell, goal_cell)
        if not path or len(path) < 2:
            print('[nav] ERROR: no path found')
            return False
        # Execute along path from second cell onward
        cur = start_cell
        for nxt in path[1:]:
            dr = nxt[0] - cur[0]
            dc = nxt[1] - cur[1]
            # Decide desired direction
            if   (dr,dc) == (-1,0): desired = 'N'
            elif (dr,dc) == (0,1):  desired = 'E'
            elif (dr,dc) == (1,0):  desired = 'S'
            elif (dr,dc) == (0,-1): desired = 'W'
            else:
                print(f'[nav] WARN: non-adjacent step {cur}->{nxt}, skipping')
                cur = nxt
                continue
            # Turn minimally to face desired
            steps = GridNavigator.turn_steps(self.current_dir, desired)
            ok = True
            if steps == 1:
                ok = self._turn_right()
            elif steps == 2:
                ok = self._turn_about()
            elif steps == 3:
                ok = self._turn_left()
            if not ok:
                print('[nav] ERROR: turn failed')
                return False
            self.current_dir = desired
            # Forward one cell
            if not self._forward_one_cell():
                print('[nav] ERROR: forward failed')
                return False
            cur = nxt
        return True


# CLI test (optional)
if __name__ == '__main__':
    print('This module is intended to be imported and used by T3_main.py')
