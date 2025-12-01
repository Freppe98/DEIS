"""slow_main.py

Deliberate, step-wise behavior controller.

Behavior outline:
 - Startup: initialize subsystems, obtain initial robot cell & world pose.
 - OPEN_PALM: Move toward gesture cell one grid cell at a time. After each move:
       - pause 2s
       - refresh world/gesture; abort if gesture type changes.
 - CLOSED_FIST: Move toward a HOME cell (first found) one grid cell at a time with same pause & refresh.
 - NONE (idle): Wander: pick a free neighboring cell, move one cell, pause 5s, refresh. Repeat until gesture changes.

Simplifications:
 - Basic Manhattan stepping rather than full A* path each iteration.
 - Approximate ticks for one-cell forward and 90-degree turns.
 - Uses raw or hysteresis gesture depending on environment variable `WP_GESTURE_NO_HYST`.
 - Heading tracked as one of N,E,S,W (0,1,2,3) starting arbitrary (N) then adjusted by turns.

Environment knobs:
 - CELL_FORWARD_TICKS (default 110)
 - TURN_90_TICKS (default 100)
 - MOVE_SPEED (default 150)
 - IDLE_WAIT_SEC (default 5)
 - STEP_WAIT_SEC (default 2)

This main is intentionally slower and more transparent for debugging early integration on the arena.
"""
from __future__ import annotations

import time
import os
from typing import Tuple, Optional, List

from redbot import start_redbot, stop_redbot, send_ticks_blocking, send_stop
from world_perception import (
    start_world_perception,
    stop_world_perception,
    get_world_state_snapshot,
    get_grid_layout,
    is_cell_free,
    find_cells_of_type,
    get_overlay_config,
)
from odometry import start_odometry, set_odometry_pose, get_odometry_snapshot
from pixel_kalman import init_kalman, kalman_predict_from_odom, kalman_update_from_world, get_kalman_state
from collision_sensor import start_collision_monitor, is_collision, cleanup as collision_cleanup
from ina260 import startINA260, stopINA260

# -------------------------------------------------------------
# Config helpers
# -------------------------------------------------------------
CELL_FORWARD_TICKS = int(os.getenv('CELL_FORWARD_TICKS','100'))
TURN_90_TICKS = int(os.getenv('TURN_90_TICKS','100'))
MOVE_SPEED = int(os.getenv('MOVE_SPEED','40'))
TURN_SPEED = int(os.getenv('TURN_SPEED','40'))
STEP_WAIT_SEC = float(os.getenv('STEP_WAIT_SEC','1'))
IDLE_WAIT_SEC = float(os.getenv('IDLE_WAIT_SEC','3'))
MAX_INIT_WAIT = float(os.getenv('SLOW_INIT_WAIT','10'))

 # Continuous motion configuration (start->monitor->stop)
USE_TICKS_MOVES = os.getenv('USE_TICKS_MOVES','0') == '1'  # set to 1 to use legacy ticks
MC_CELL_TOL_PX = float(os.getenv('MC_CELL_TOL_PX','20'))
MC_MAX_FWD_SEC = float(os.getenv('MC_MAX_FWD_SEC','6.0'))
MC_MAX_TURN_SEC = float(os.getenv('MC_MAX_TURN_SEC','6.0'))
MC_HEADING_TOL_DEG = float(os.getenv('MC_HEADING_TOL_DEG','5.0'))
MC_TURN_SOURCE = os.getenv('MC_TURN_SOURCE','WORLD')  # WORLD or KALMAN
TURN_WITH_TICKS = os.getenv('TURN_WITH_TICKS','1') == '1'  # default: use ticks-based turning
PX_PER_TICK = float(os.getenv('PX_PER_TICK','0'))  # optional calibration: pixels per encoder tick (forward)

# -------------------------------------------------------------
# Debug helpers
# -------------------------------------------------------------

def debug_mapping_context(label: str, current: Tuple[int,int], target: Tuple[int,int]):
    """Print key overlay/grid mapping info and cell centers to verify alignment."""
    overlay = get_overlay_config()
    if not overlay:
        print(f"[MAP.debug] {label}: overlay not loaded")
        return
    b = overlay.get('arena_bounds', {})
    g = overlay.get('grid', {})
    rw = overlay.get('real_world', {})
    cw = g.get('cell_size_px', {}).get('x', 1)
    ch = g.get('cell_size_px', {}).get('y', 1)
    left, top = b.get('left'), b.get('top')
    mmx, mmy = rw.get('mm_per_pixel_x'), rw.get('mm_per_pixel_y')
    ox, oy = rw.get('origin_mm', {}).get('x'), rw.get('origin_mm', {}).get('y')
    def center_px(cell):
        r, c = cell
        u = left + (c + 0.5) * cw
        v = top + (r + 0.5) * ch
        return (u, v)
    cu, cv = center_px(current)
    tu, tv = center_px(target)
    print(f"[MAP.ctx] {label}: bounds=({left},{top})-({b.get('right')},{b.get('bottom')}), rows={g.get('rows')} cols={g.get('cols')} cell_px=({cw},{ch})")
    print(f"[MAP.ctx] origin_mm=({ox},{oy}) mm_per=({mmx:.4f},{mmy:.4f})")
    print(f"[MAP.ctx] current_cell={current} center_px=({cu:.1f},{cv:.1f}) target_cell={target} center_px=({tu:.1f},{tv:.1f})")

# -------------------------------------------------------------
# Heading utilities
# -------------------------------------------------------------
HEADING_DIRS = ['N','E','S','W']  # 0..3
# Map heading to delta row/col when moving forward (row increases downward)
HEADING_DELTAS = {
    0: (-1, 0),  # N
    1: (0, 1),   # E
    2: (1, 0),   # S
    3: (0, -1),  # W
}

def rotate_heading(curr: int, target: int) -> List[int]:
    """Return sequence of relative 90° turns (each is +1 or -1) to get from curr to target.
    We choose the shorter rotation direction.
    +1 means turn right (clockwise), -1 means left (counter-clockwise).
    """
    diff = (target - curr) % 4
    if diff == 0:
        return []
    if diff == 3:  # equivalent to -1
        return [-1]
    if diff == 1:
        return [+1]
    if diff == 2:
        return [+1, +1]  # two right turns
    # Should not reach here
    return []

# -------------------------------------------------------------
# Movement primitives (approximate ticks)
# -------------------------------------------------------------

def do_turn(direction: int) -> bool:
    """Perform a single 90° turn; direction +1 right / -1 left."""
    # For an in-place turn, left wheel forward, right wheel backward or vice versa.
    ticks = TURN_90_TICKS
    if direction == +1:
        return send_ticks_blocking(ticks, ticks, TURN_SPEED, -TURN_SPEED, timeout=8.0)
    else:
        return send_ticks_blocking(ticks, ticks, -TURN_SPEED, TURN_SPEED, timeout=8.0)


def do_forward_one_cell() -> bool:
    return send_ticks_blocking(CELL_FORWARD_TICKS, CELL_FORWARD_TICKS, MOVE_SPEED, MOVE_SPEED, timeout=12.0)

# -------------------------------------------------------------
# Continuous motion primitives (start/monitor/stop)
# -------------------------------------------------------------

def _cell_center_px(cell: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    overlay = get_overlay_config()
    if not overlay:
        return None
    b = overlay.get('arena_bounds', {})
    g = overlay.get('grid', {})
    cw = g.get('cell_size_px',{}).get('x',1)
    ch = g.get('cell_size_px',{}).get('y',1)
    left, top = b.get('left'), b.get('top')
    r, c = cell
    try:
        u = left + (c + 0.5) * cw
        v = top + (r + 0.5) * ch
        return (float(u), float(v))
    except Exception:
        return None


def _get_world_heading_deg(ws) -> Optional[float]:
    if MC_TURN_SOURCE.upper() == 'WORLD' and ws.robot_heading_deg is not None:
        return float(ws.robot_heading_deg)
    try:
        from math import degrees
        _, _, th = get_kalman_state()
        return float(degrees(th))
    except Exception:
        return None


def _angle_within_card(ang_deg: float, card: str, tol: float) -> bool:
    targets = {'E': 0.0, 'S': 90.0, 'W': 180.0, 'N': -90.0}
    if card not in targets:
        return False
    tgt = targets[card]
    diff = ang_deg - tgt
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0
    return abs(diff) <= tol


def _start_turn_nonblocking(direction: int):
    from redbot import send_ticks
    ticks = 100000
    if direction == +1:
        send_ticks(ticks, ticks, MOVE_SPEED, -MOVE_SPEED)
    else:
        send_ticks(ticks, ticks, -MOVE_SPEED, MOVE_SPEED)


def _continuous_turn_to_card(current_card: str, target_card: str) -> bool:
    idx = {'N':0,'E':1,'S':2,'W':3}
    curr_i = idx.get(current_card, 0)
    tgt_i = idx.get(target_card, curr_i)
    turns = rotate_heading(curr_i, tgt_i)
    turn_dir = +1 if (turns and turns[0] == +1) else (-1 if (turns and turns[0] == -1) else 0)
    if turn_dir == 0:
        return True
    _start_turn_nonblocking(turn_dir)
    t0 = time.time()
    ok = False
    last_ang = None
    while True:
        ws = get_world_state_snapshot()
        ang = _get_world_heading_deg(ws)
        if ang is not None:
            last_ang = ang
            if _angle_within_card(ang, target_card, MC_HEADING_TOL_DEG):
                ok = True
                break
        if (time.time() - t0) > MC_MAX_TURN_SEC:
            break
        time.sleep(0.05)
    try:
        send_stop()
    except Exception:
        pass
    if last_ang is not None:
        print(f"[TURN.cont] target={target_card} last_ang={last_ang:.1f} tol={MC_HEADING_TOL_DEG} ok={ok}")
    else:
        print(f"[TURN.cont] target={target_card} no heading samples ok={ok}")
    return ok


def _ticks_turn_to_card(target_card: str) -> bool:
    """Turn toward target cardinal using small tick bursts guided by heading.
    Falls back to 90° ticks if heading unavailable.
    """
    # Map card to target angle in degrees (image frame)
    card_to_deg = {'E': 0.0, 'S': 90.0, 'W': 180.0, 'N': -90.0}
    tgt = card_to_deg.get(target_card)
    if tgt is None:
        return False
    t0 = time.time()
    # Helper to get current angle
    def get_ang():
        ws = get_world_state_snapshot()
        ang = _get_world_heading_deg(ws)
        return ang
    ang = get_ang()
    if ang is None:
        # Fallback: perform 90° turns via do_turn toward desired heading
        # Determine current cardinal from last known or assume heading index
        ws0 = get_world_state_snapshot()
        cur_card = ws0.robot_heading_card if ws0.robot_heading_card else None
        if cur_card is None:
            return True  # nothing better to do; skip
        idx = {'N':0,'E':1,'S':2,'W':3}
        curr_i = idx.get(cur_card,0)
        tgt_i = idx.get(target_card,curr_i)
        for cmd in rotate_heading(curr_i, tgt_i):
            ok = do_turn(cmd)
            if not ok:
                return False
            time.sleep(0.1)
        return True
    # Guided by heading
    deg_per_tick90 = float(TURN_90_TICKS) / 90.0
    ok = False
    while True:
        # compute normalized error [-180,180]
        err = ang - tgt
        while err > 180.0:
            err -= 360.0
        while err < -180.0:
            err += 360.0
        if abs(err) <= MC_HEADING_TOL_DEG:
            ok = True
            break
        if (time.time() - t0) > MC_MAX_TURN_SEC:
            break
        # choose a small burst toward reducing error
        step_deg = min(20.0, max(5.0, abs(err) * 0.5))
        ticks = max(3, int(round(deg_per_tick90 * step_deg)))
        if err > 0:  # current > target -> need to rotate left (ccw)
            # left turn: left wheel backward, right forward
            ok_burst = send_ticks_blocking(ticks, ticks, -TURN_SPEED, TURN_SPEED, timeout=2.0)
        else:
            # right turn: left forward, right backward
            ok_burst = send_ticks_blocking(ticks, ticks, TURN_SPEED, -TURN_SPEED, timeout=2.0)
        if not ok_burst:
            break
        time.sleep(0.02)
        ang = get_ang()
        if ang is None:
            # if measurement dropped, continue a small additional burst and then exit
            continue
    return ok




def _continuous_forward_to_cell(current: Tuple[int,int], expected_next: Tuple[int,int]) -> bool:
    from redbot import send_ticks
    send_ticks(100000, 100000, MOVE_SPEED, MOVE_SPEED)
    t0 = time.time()
    ok = False
    unexpected = False
    tgt_px = _cell_center_px(expected_next)
    last_ws = None
    while True:
        ws = get_world_state_snapshot()
        last_ws = ws
        rc = ws.robot_cell
        if rc is not None:
            if rc == expected_next:
                ok = True
                break
            if rc != current:
                unexpected = True
                break
        if ws.robot_pixel and tgt_px is not None:
            du = ws.robot_pixel[0] - tgt_px[0]
            dv = ws.robot_pixel[1] - tgt_px[1]
            dist = (du*du + dv*dv) ** 0.5
            if dist <= MC_CELL_TOL_PX:
                ok = True
                break
        if (time.time() - t0) > MC_MAX_FWD_SEC:
            break
        time.sleep(0.06)
    try:
        send_stop()
    except Exception:
        pass
    if unexpected:
        curr = last_ws.robot_cell if last_ws else None
        print(f"[MOVE.cont] UNEXPECTED CELL: wanted={expected_next} current={curr}")
    print(f"[MOVE.cont] forward to {expected_next} ok={ok} elapsed={(time.time()-t0):.2f}s")
    return ok and not unexpected

# -------------------------------------------------------------
# Pixel-calibrated forward (ticks from pixel distance)
# -------------------------------------------------------------

def _cell_center_px(cell: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    overlay = get_overlay_config()
    if not overlay:
        return None
    b = overlay.get('arena_bounds', {})
    g = overlay.get('grid', {})
    cw = g.get('cell_size_px',{}).get('x',1)
    ch = g.get('cell_size_px',{}).get('y',1)
    left, top = b.get('left'), b.get('top')
    r, c = cell
    try:
        u = left + (c + 0.5) * cw
        v = top + (r + 0.5) * ch
        return (float(u), float(v))
    except Exception:
        return None


def _forward_to_target_center_by_ticks(current_cell: Tuple[int,int], next_cell: Tuple[int,int]) -> bool:
    """Compute pixel distance from current cell center to next cell center and drive forward using calibrated PX_PER_TICK.
    Falls back to legacy one-cell ticks if calibration not available.
    """
    if PX_PER_TICK <= 0:
        return do_forward_one_cell()
    cur = _cell_center_px(current_cell)
    nxt = _cell_center_px(next_cell)
    if cur is None or nxt is None:
        return do_forward_one_cell()
    # distance along heading axis; we assume manhattan step so axis-aligned
    du = nxt[0] - cur[0]
    dv = nxt[1] - cur[1]
    dist_px = abs(du) + abs(dv)
    ticks = int(round(dist_px / PX_PER_TICK))
    if ticks <= 0:
        return True
    return send_ticks_blocking(ticks, ticks, MOVE_SPEED, MOVE_SPEED, timeout=12.0)

# -------------------------------------------------------------
# Target selection helpers
# -------------------------------------------------------------

def manhattan_step(from_cell: Tuple[int,int], to_cell: Tuple[int,int]) -> Tuple[int,int]:
    fr, fc = from_cell
    tr, tc = to_cell
    if fr == tr and fc == tc:
        return from_cell
    # Prioritize row movement then column
    if fr < tr:
        return (fr+1, fc)
    if fr > tr:
        return (fr-1, fc)
    if fc < tc:
        return (fr, fc+1)
    if fc > tc:
        return (fr, fc-1)
    return from_cell


def pick_idle_neighbor(cell: Tuple[int,int], grid: List[List[str]]) -> Tuple[int,int]:
    r, c = cell
    candidates = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
    free = [xy for xy in candidates if 0 <= xy[0] < len(grid) and 0 <= xy[1] < len(grid[0]) and is_cell_free(xy[0], xy[1])]
    if not free:
        return cell
    # Simple cycle: pick first free that isn't current
    for f in free:
        if f != cell:
            return f
    return cell

# -------------------------------------------------------------
# World initialization
# -------------------------------------------------------------

def wait_initial_world(timeout: float) -> Optional[Tuple[int,int]]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        ws = get_world_state_snapshot()
        if ws.robot_visible and ws.robot_cell:
            return ws.robot_cell
        time.sleep(0.3)
    return None

# -------------------------------------------------------------
# Fallback: estimate cell from Kalman + overlay
# -------------------------------------------------------------

def estimate_cell_from_kalman() -> Optional[Tuple[int,int]]:
    overlay = get_overlay_config()
    if not overlay:
        return None
    try:
        u_px, v_px, _ = get_kalman_state()
        rw = overlay['real_world']
        bounds = overlay['arena_bounds']
        grid = overlay['grid']
        # Already in arena pixel space
        u = float(u_px)
        v = float(v_px)

        # Map pixel -> cell
        cell_w = grid['cell_size_px']['x']
        cell_h = grid['cell_size_px']['y']
        col = int((u - bounds['left']) // cell_w)
        row = int((v - bounds['top']) // cell_h)
        if 0 <= row < grid['rows'] and 0 <= col < grid['cols']:
            return (row, col)
    except Exception:
        return None
    return None

# -------------------------------------------------------------
# Behavior loops
# -------------------------------------------------------------

def step_move_towards(current: Tuple[int,int], target: Tuple[int,int], heading: int) -> int:
    """Execute one forward cell toward target from current cell, including heading turns.
    Returns updated heading.
    """
    if current is None:
        return heading
    if current == target:
        return heading
    next_cell = manhattan_step(current, target)
    dr = next_cell[0] - current[0]
    dc = next_cell[1] - current[1]
    print(f"[STEP.ctx] current={current} target={target} next={next_cell} delta=(dr={dr},dc={dc}) heading={HEADING_DIRS[heading]}")
    # Determine desired heading index
    desired = None
    for h, (hr, hc) in HEADING_DELTAS.items():
        if (hr, hc) == (dr, dc):
            desired = h
            break
    if desired is None:
        print(f"[STEP] Cannot determine heading for delta {(dr,dc)}")
        return heading
    print(f"[STEP.ctx] desired_heading={HEADING_DIRS[desired]} turns={rotate_heading(heading, desired)}")

    if USE_TICKS_MOVES:
        # Legacy ticks-based turning and forward
        for turn_cmd in rotate_heading(heading, desired):
            ok = do_turn(turn_cmd)
            print(f"[TURN] {'right' if turn_cmd==1 else 'left'} ok={ok}")
            time.sleep(0.5)
        heading = desired
        ok = do_forward_one_cell()
        print(f"[MOVE] forward one cell from {current} toward {target} ok={ok}")
        return heading
    else:
        # Continuous perception-driven turning and forward
        ws0 = get_world_state_snapshot()
        cur_card = ws0.robot_heading_card if (os.getenv('SLOW_USE_WORLD_HEADING','0')=='1' and ws0.robot_heading_card) else HEADING_DIRS[heading]
        tgt_card = HEADING_DIRS[desired]
        if TURN_WITH_TICKS:
            ok_turn = _ticks_turn_to_card(tgt_card)
            print(f"[TURN.ticks] -> {tgt_card} ok={ok_turn}")
        else:
            ok_turn = _continuous_turn_to_card(cur_card, tgt_card)
            print(f"[TURN.cont] {cur_card}->{tgt_card} ok={ok_turn}")
            heading = desired  # adopt desired heading after turn
            # Prefer pixel-calibrated ticks if available, else continuous forward monitor
            ok_move = _forward_to_target_center_by_ticks(current, next_cell) if PX_PER_TICK > 0 else _continuous_forward_to_cell(current, next_cell)
        print(f"[MOVE.cont] {current}->{next_cell} ok={ok_move}")
        return heading


def behavior_loop():
    heading = 0  # start facing North arbitrarily
    cached_cell: Optional[Tuple[int,int]] = None
    current_mode: Optional[str] = None  # 'PALM' | 'FIST' | 'IDLE'
    print("[SLOW] Entering behavior loop (slow mode).")
    while True:
        ws = get_world_state_snapshot()
        # Pixel-kalman fusion: predict from odometry, then update from perception
        try:
            od = get_odometry_snapshot()
            overlay_cfg = get_overlay_config()
            kalman_predict_from_odom(od, 1.0/30.0, overlay_cfg)
        except Exception:
            pass
        try:
            kalman_update_from_world(ws)
        except Exception:
            pass
        grid = get_grid_layout()
        # Optionally update heading from world perception heading
        if os.getenv('SLOW_USE_WORLD_HEADING','0') == '1' and ws.robot_heading_card:
            try:
                card_to_idx = {'N':0,'E':1,'S':2,'W':3}
                nh = card_to_idx.get(ws.robot_heading_card, heading)
                if nh != heading:
                    print(f"[SLOW.head] world heading card={ws.robot_heading_card} idx={nh}")
                heading = nh
            except Exception:
                pass
        # Update cached robot cell when visible
        if ws.robot_cell:
            cached_cell = ws.robot_cell
        # If no cached cell yet, attempt fallback from Kalman
        if cached_cell is None:
            est = estimate_cell_from_kalman()
            if est is not None:
                cached_cell = est

        # Require grid loaded; allow movement using cached_cell when robot not visible
        if not grid or (ws.robot_cell is None and cached_cell is None):
            print("[SLOW] Waiting for grid/robot_cell...")
            time.sleep(1.0)
            continue

        gesture = ws.gesture_type
        robot_cell = ws.robot_cell or cached_cell

        if gesture == 'OPEN_PALM' and ws.gesture_cell:
            target = ws.gesture_cell
            if current_mode != 'PALM':
                print(f"[PALM.enter] robot_cell={robot_cell} robot_px={ws.robot_pixel} gesture_cell={target} gesture_px={ws.gesture_pixel}")
                current_mode = 'PALM'
            debug_mapping_context('PALM', robot_cell, target)
            # Continue stepping while gesture remains OPEN_PALM and target visible
            while True:
                ws2 = get_world_state_snapshot()
                if ws2.gesture_type != 'OPEN_PALM' or not ws2.gesture_cell:
                    print("[PALM.exit] Gesture changed/vanished")
                    current_mode = None
                    break
                current2 = ws2.robot_cell or cached_cell
                if current2 is None:
                    print("[PALM.exit] No current cell; abort")
                    current_mode = None
                    break
                if current2 == ws2.gesture_cell:
                    print("[PALM.exit] Reached gesture cell")
                    current_mode = None
                    break
                heading = step_move_towards(current2, ws2.gesture_cell, heading)
                # Minimal pause message
                print("[PALM] step complete; refreshing")
                time.sleep(STEP_WAIT_SEC)
            continue

        elif gesture == 'CLOSED_FIST':
            homes = find_cells_of_type('H')
            target = homes[0] if homes else robot_cell
            if current_mode != 'FIST':
                print(f"[FIST.enter] robot_cell={robot_cell} robot_px={ws.robot_pixel} home_cell={target}")
                current_mode = 'FIST'
            while True:
                ws2 = get_world_state_snapshot()
                if ws2.gesture_type != 'CLOSED_FIST':
                    print("[FIST.exit] Gesture changed")
                    current_mode = None
                    break
                current2 = ws2.robot_cell or cached_cell
                if current2 is None:
                    print("[FIST.exit] No current cell; abort")
                    current_mode = None
                    break
                if current2 == target:
                    print("[FIST.exit] Reached home cell")
                    current_mode = None
                    break
                heading = step_move_towards(current2, target, heading)
                print("[FIST] step complete; refreshing")
                time.sleep(STEP_WAIT_SEC)
            continue

        else:
            # Idle wandering
            neighbor = pick_idle_neighbor(robot_cell, grid)
            if neighbor == robot_cell:
                if current_mode != 'IDLE':
                    print(f"[IDLE.enter] robot_cell={robot_cell} robot_px={ws.robot_pixel}")
                    current_mode = 'IDLE'
                print("[IDLE] No free neighbor; staying put")
                time.sleep(IDLE_WAIT_SEC)
                continue
            print(f"[IDLE] Wander to {neighbor}")
            debug_mapping_context('IDLE', robot_cell, neighbor)
            heading = step_move_towards(robot_cell, neighbor, heading)
            print("[IDLE] step complete; pausing")
            time.sleep(IDLE_WAIT_SEC)

        # Reactive collision stop
        if is_collision():
            print('[SAFETY] Collision detected; issuing stop.')
            try:
                send_stop()
            except Exception:
                pass
            time.sleep(1.0)

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    print('[SLOW] Startup sequence...')
    start_redbot()
    start_odometry(wheel_radius=0.0325, ticks_per_rev=170, wheel_base=0.157, loop_hz=30.0)
    start_world_perception('gps_overlay.json', 'grid_layout.json', loop_hz=10.0)
    start_collision_monitor()
    startINA260()

    # Print cell pixel size and arena bounds for quick verification
    try:
        ov = get_overlay_config()
        if ov:
            g = ov.get('grid', {})
            b = ov.get('arena_bounds', {})
            cw = g.get('cell_size_px',{}).get('x')
            ch = g.get('cell_size_px',{}).get('y')
            rows = g.get('rows')
            cols = g.get('cols')
            #print(f"[MAP.info] cell_px=({cw},{ch}) rows={rows} cols={cols} bounds=({b.get('left')},{b.get('top')})-({b.get('right')},{b.get('bottom')})")
    except Exception:
        pass

    print('[SLOW] Waiting for initial perception measurement (pixel)...')
    # Block until world_perception provides a robot_pixel; no fallback to (0,0,0)
    while True:
        ws0 = get_world_state_snapshot()
        if ws0.robot_pixel:
            break
        time.sleep(0.2)
    u0, v0 = ws0.robot_pixel
    init_kalman(u0, v0, 0.0)
    # Also seed odometry pose in meters if overlay provides mapping
    try:
        ov = get_overlay_config()
        if ov and 'real_world' in ov and 'arena_bounds' in ov:
            rw = ov['real_world']
            b = ov['arena_bounds']
            x_mm = rw['origin_mm']['x'] + (u0 - b['left']) * rw['mm_per_pixel_x']
            y_mm = rw['origin_mm']['y'] + (v0 - b['top']) * rw['mm_per_pixel_y']
            set_odometry_pose(x_mm/1000.0, y_mm/1000.0, 0.0)
            print(f"[SLOW] Init from perception: px=({u0:.1f},{v0:.1f}) mm=({x_mm:.1f},{y_mm:.1f})")
        else:
            print(f"[SLOW] Init from perception: px=({u0:.1f},{v0:.1f}) (overlay mm unavailable)")
    except Exception:
        print(f"[SLOW] Init from perception: px=({u0:.1f},{v0:.1f}); odometry seed skipped")

    # Optionally seed initial heading from world perception
    if os.getenv('SLOW_INIT_HEADING_FROM_WORLD','0') == '1':
        ws0 = get_world_state_snapshot()
        card = ws0.robot_heading_card
        if card:
            card_to_idx = {'N':0,'E':1,'S':2,'W':3}
            idx = card_to_idx.get(card, 0)
            print(f"[SLOW] Initial heading from world: {card} -> idx={idx}")
            # We don't store heading globally; behavior_loop prints and tracks

    # Optionally disable gesture hysteresis for immediacy
    if os.getenv('WP_GESTURE_NO_HYST','0') == '1':
        print('[SLOW] Using raw gesture types (hysteresis bypass).')

    try:
        behavior_loop()
    except KeyboardInterrupt:
        print('\n[SLOW] KeyboardInterrupt; shutting down.')
    finally:
        try: stop_world_perception()
        except Exception: pass
        try: collision_cleanup()
        except Exception: pass
        stopINA260()
        stop_redbot()
        print('[SLOW] Exit complete.')

if __name__ == '__main__':
    main()
