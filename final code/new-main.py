#!/usr/bin/env python3
"""
TOLLGATE3/new-main.py

Idea:
- Plan on GRID cells with A*
- Execute bounded actions with ticks (TURN or FORWARD)

Dependencies:
- world_perception.py (minimal): start_perception(), get_snapshot(), get_grid(), get_home_cell(), stop_perception()
- astar-api.py: search(), next_action_from_path()
- redbot.py: start_redbot(), send_ticks_blocking(), send_stop(), stop_redbot()
- collision_sensor.py: start_collision_monitor(), is_collision(), cleanup()
- ina260.py: startINA260(), getVoltage(), getCurrent(), stopINA260()

"""

from __future__ import annotations

import os
import time
import math
import copy
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, List

from redbot import start_redbot, stop_redbot, send_ticks_blocking, send_stop, send_ticks, is_move_done
from collision_sensor import start_collision_monitor, is_collision, cleanup as collision_cleanup
from ina260 import startINA260, stopINA260
from world_perception import start_perception, stop_perception, get_snapshot, get_grid, get_home_cell, rc_is_neighbor_or_same, rc_neighbors8, rc_is_4neighbor_or_same, rc_distance_mm, rc_distance_px


# --------------------------
# Config
# --------------------------

OVERLAY_JSON = os.getenv("OVERLAY_JSON", "./gps_overlay.json")
GRID_JSON    = os.getenv("GRID_JSON", "./grid.json")

ROBOT_POS_ID  = int(os.getenv("ROBOT_POS_ID", "0"))
ROBOT_HEAD_ID = int(os.getenv("ROBOT_HEAD_ID", "1"))

TURN_90_TICKS       = int(os.getenv("TURN_90_TICKS", "95"))
CELL_FORWARD_TICKS  = int(os.getenv("CELL_FORWARD_TICKS", "75"))

MOVE_SPEED = int(os.getenv("MOVE_SPEED", "50"))
TURN_SPEED = int(os.getenv("TURN_SPEED", "50")) # minimum 40, noticed at least 50 is needed when all 3d parts are mounted.

HEADING_TOL_DEG = float(os.getenv("HEADING_TOL_DEG", "5.0"))
SETTLE_SEC      = float(os.getenv("SETTLE_SEC", "2"))

STABLE_SAMPLES  = int(os.getenv("STABLE_SAMPLES", "2"))
STEP_TIMEOUT    = float(os.getenv("STEP_TIMEOUT", "10.0"))

# Number of actions (TURN/FORWARD) to execute per planning cycle
EXEC_CHUNK_ACTIONS = int(os.getenv("EXEC_CHUNK_ACTIONS", "5"))

# Safety / recovery (How many ticks to drive backwards if collision)
BACKUP_TICKS = int(os.getenv("BACKUP_TICKS", "40"))

# Wheel/heading calibration
FORWARD_L_SIGN = int(os.getenv("FORWARD_L_SIGN", "1"))   # +1 or -1
FORWARD_R_SIGN = int(os.getenv("FORWARD_R_SIGN", "1"))   # +1 or -1
TURN_SIGN      = int(os.getenv("TURN_SIGN", "-1"))        # +1 normal, -1 invert CCW/CW mapping
HEADING_OFFSET_DEG = float(os.getenv("HEADING_OFFSET_DEG", "0"))

# Goal acceptance tolerance (rectangular, in cells)
GOAL_TOL_ROWS = int(os.getenv("GOAL_TOL_ROWS", "4"))  # RETIRED
GOAL_TOL_COLS = int(os.getenv("GOAL_TOL_COLS", "4"))  # RETIRED

# Pixel-distance goal tolerance (rectified pixels)
GOAL_PX_TOL = int(os.getenv("GOAL_PX_TOL", "100"))  # PREFFERED


# --------------------------
# A* dynamic loader (file name has hyphen)
# --------------------------

def _load_astar_api() :
    base = Path(__file__).resolve().parent
    astar_path = base / "astar-api.py"
    spec = importlib.util.spec_from_file_location("mi_astar_api", str(astar_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {astar_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

ASTAR = _load_astar_api()


# --------------------------
# Util
# --------------------

def wrap_err_deg(target: float, curr: float) -> float:
    """Shortest signed error in degrees, range [-180, +180)."""
    return (target - curr + 180.0) % 360.0 - 180.0

def heading_card_after_turn(card: str, turn: str) -> str:
    right = {"N":"E","E":"S","S":"W","W":"N"}
    left  = {"N":"W","W":"S","S":"E","E":"N"}
    if turn == "RIGHT":
        return right[card]
    if turn == "LEFT":
        return left[card]
    if turn == "AROUND":
        return right[right[card]]
    return card

def card_to_target_deg(card: str) -> float:
    """
    Keep consistent with minimal world_perception:
    0=E, 90=S, 180=W, 270=N.
    """
    m = {"E":0.0, "S":90.0, "W":180.0, "N":270.0}
    return m.get(card, 0.0)

def build_astar_grid(layout_grid: List[List[int]]) -> List[List[int]]:
    """
    Convert layout grid (0=free,1=obstacle,2=home) into A* grid:
    - free: 0
    - obstacle: 1
    - home: 4
    """
    g = []
    for row in layout_grid:
        out = []
        for v in row:
            if int(v) == 1:
                out.append(1)
            elif int(v) == 2:
                out.append(4)
            else:
                out.append(0)
        g.append(out)
    return g

def cardinalize_path(path_xy: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    Ensure all consecutive steps are 4-neighbor (no diagonals).
    If a diagonal (dx!=0 and dy!=0) is found between (x0,y0)->(x1,y1),
    insert an intermediate node to break it into two cardinal moves:
    preference: move rows first, then cols.
    """
    if not path_xy or len(path_xy) < 2:
        return path_xy
    out: List[Tuple[int,int]] = [path_xy[0]]
    for i in range(1, len(path_xy)):
        x0, y0 = out[-1]
        x1, y1 = path_xy[i]
        dx = x1 - x0
        dy = y1 - y0
        # Break diagonals by inserting row-first intermediate
        if dx != 0 and dy != 0:
            step_y = 1 if dy > 0 else -1
            mid = (x0, y0 + step_y)
            out.append(mid)
            out.append((x1, y1))
            continue
        # Split long cardinal jumps into unit steps
        if dx == 0 and abs(dy) > 1:
            step_y = 1 if dy > 0 else -1
            for _ in range(abs(dy)):
                y0 += step_y
                out.append((x0, y0))
            continue
        if dy == 0 and abs(dx) > 1:
            step_x = 1 if dx > 0 else -1
            for _ in range(abs(dx)):
                x0 += step_x
                out.append((x0, y0))
            continue
        out.append((x1, y1))
    return out

def rc_within_rect_tol(a: Tuple[int,int], b: Tuple[int,int], tol_r: int = GOAL_TOL_ROWS, tol_c: int = GOAL_TOL_COLS) -> bool:
    if a is None or b is None:
        return False
    ar, ac = a
    br, bc = b
    return abs(ar - br) <= int(tol_r) and abs(ac - bc) <= int(tol_c)

def set_start_goal(astar_grid: List[List[int]],
                   start_cell_rc: Tuple[int,int],
                   goal_cell_rc: Tuple[int,int],
                   goaltype: str) -> List[List[int]]:
    """
    A* uses (x,y) internally but reads grid as grid[y][x].
    cells are (row,col) = (y,x).
    """
    g = copy.deepcopy(astar_grid)
    sr, sc = start_cell_rc
    gr, gc = goal_cell_rc
    # mark start
    g[sr][sc] = 2
    # mark goal
    g[gr][gc] = 4 if goaltype.upper() == "HOME" else 3
    return g

def wait_stable_robot_cell(max_wait_s: float = 20.0) -> Tuple[int,int]:
    t0 = time.time()
    last: Optional[Tuple[int,int]] = None
    same = 0
    while (time.time() - t0) < max_wait_s:
        ws = get_snapshot()
        if ws.robot_cell is None:
            same = 0
            time.sleep(0.1)
            continue
        # Neighbor acceptance for stability: accept same or 8-neighbor
        if last is not None and rc_is_neighbor_or_same(ws.robot_cell, last):
            same += 1
        else:
            same = 1
            last = ws.robot_cell
        if same >= STABLE_SAMPLES:
            return ws.robot_cell
        time.sleep(0.1)
    raise TimeoutError("Robot cell not stable / not visible.")

def wait_goal_cell(max_wait_s: float = 60.0) -> Tuple[str, Tuple[int,int]]:
    """
    Gesture mapping:
    - OPEN_PALM -> FOOD target
    - CLOSED_FIST -> HOME target
    """
    t0 = time.time()
    last_goal = None
    same = 0
    while (time.time() - t0) < max_wait_s:
        ws = get_snapshot()
        goaltype = None
        goal = None
        print(ws.gesture_type, ws.gesture_cell)
        if ws.gesture_type == "OPEN_PALM" and ws.gesture_cell is not None:
            goaltype, goal = "FOOD", ws.gesture_cell
        elif ws.gesture_type == "CLOSED_FIST":
            hc = ws.home_cell
            if hc is not None:
                goaltype, goal = "HOME", hc

        if goal is None:
            same = 0
            time.sleep(0.1)
            continue

        cur = (goaltype, goal)
        if cur == last_goal:
            same += 1
        else:
            same = 1
            last_goal = cur

        if same >= STABLE_SAMPLES:
            return goaltype, goal

        time.sleep(0.1)

    raise TimeoutError("No stable goal gesture seen.")


# --------------------------
# Motion (tick-bounded)
# --------------------------

def wait_done_or_abort(timeout: Optional[float], goal_rc: Optional[Tuple[int,int]] = None) -> bool:
    """Wait until redbot motion DONE, collision abort, pixel-goal reached, or timeout.
    Returns True on DONE, False on abort/timeout/goal-reached (mid-motion stop).
    """
    deadline = None if timeout is None else (time.time() + float(timeout))
    while True:

        # Mid-motion pixel goal stop to avoid overshoot
        if goal_rc is not None:
            ws_goal = get_snapshot()
            #print("3MM DISTANCE: ", rc_distance_mm(ws_goal.robot_cell, goal_rc))
            plz_goal = rc_distance_px(ws_goal.robot_cell, goal_rc)
            #print("3PX DISTANCE: ", plz_goal)
            if plz_goal < GOAL_PX_TOL:
                send_stop()
                print("move false (goal reached mid-motion)")
                return False


        if is_move_done():
            print("move done")
            return True
        # Collision abort
        if is_collision():
            try:
                send_stop()
            except Exception:
                pass
            print("move false")
            return False

        # Timeout abort
        if deadline is not None and time.time() >= deadline:
            try:
                send_stop()
            except Exception:
                pass
            print("move false")
            return False
        time.sleep(0.02)

def turn_to_heading_exact(target_deg: float,
                          tol_deg: float = HEADING_TOL_DEG,
                          max_corrections: int = 2) -> bool:
    """
    Uses WORLD heading (from perception snapshot), but only in bounded steps:
    command exact ticks for current error, then settle and re-check.
    """
    print("TURN EXACT")

    ticks_per_deg = float(TURN_90_TICKS) / 90.0

    for _ in range(1 + max_corrections):
        ws = get_snapshot()
        if ws.robot_heading_deg is None:
            print("Heading not available")
            # If heading not available, fallback to blind 90/180 turns elsewhere
            return False

        curr = ws.robot_heading_deg
        print("CURRENT HEAD: ", curr, "TARGET HEAD: ", target_deg)
        if curr is not None:
            curr = (curr + HEADING_OFFSET_DEG) % 360.0
        err = wrap_err_deg(target_deg, curr)

        if abs(err) <= tol_deg:
            return True

        ticks = int(round(abs(err) * ticks_per_deg))
        ticks = max(3, ticks)

        # err > 0 means CCW/left
        l_spd = -TURN_SPEED if err > 0 else +TURN_SPEED
        r_spd = +TURN_SPEED if err > 0 else -TURN_SPEED
        if TURN_SIGN == -1:
            l_spd, r_spd = -l_spd, -r_spd

        print(f"[STEP] TURN err={err:.1f}deg ticks={ticks} lspd={l_spd} rspd={r_spd}")
        send_ticks(ticks, ticks, l_spd, r_spd)
        ok = wait_done_or_abort(STEP_TIMEOUT)

        if not ok:
            return False

        time.sleep(SETTLE_SEC)

        # DEBUG..... Measure post-step error to classify over/under-shoot
        ws2 = get_snapshot()
        if ws2.robot_heading_deg is not None:
            curr2 = (ws2.robot_heading_deg + HEADING_OFFSET_DEG) % 360.0
            post_err = wrap_err_deg(target_deg, curr2)
            flipped = (err > 0 and post_err < 0) or (err < 0 and post_err > 0)
            if flipped:
                print(f"[TURN] overshoot by {abs(post_err):.1f}deg (post err={post_err:.1f})")
            elif abs(post_err) > abs(err):
                # moved away from target without crossing
                print(f"[TURN] undershoot grew to {abs(post_err):.1f}deg (post err={post_err:.1f})")
            else:
                print(f"[TURN] residual error {abs(post_err):.1f}deg (post err={post_err:.1f})")

    # final check
    ws = get_snapshot()
    if ws.robot_heading_deg is None:
        return False
    return abs(wrap_err_deg(target_deg, ws.robot_heading_deg)) <= tol_deg


def backup_small() -> None:
    try:
        send_ticks(BACKUP_TICKS, BACKUP_TICKS, -MOVE_SPEED, -MOVE_SPEED)
        wait_done_or_abort(3.0)
        time.sleep(SETTLE_SEC)
    except Exception:
        pass


# --------------------------
# Planner + executor
# --------------------------

def plan_path(start_rc: Tuple[int,int], goal_rc: Tuple[int,int], goaltype: str) -> Optional[List[Tuple[int,int]]]:
    layout = get_grid()
    if not layout:
        return None

    base_grid = build_astar_grid(layout)
    g = set_start_goal(base_grid, start_rc, goal_rc, goaltype)

    nodes, path_deque, *_ = ASTAR.search(g, goaltype=goaltype)
    if path_deque is None:
        return None
    # A* returns [(x,y), ...]
    path = list(path_deque)
    # clean to cardinal-only moves to match executor constraints
    return cardinalize_path(path)



def execute_all(goal_rc: Tuple[int,int], goaltype: str) -> bool:
    """
    full-path executor with exact heading turns (Can be launched in CHUNKS).
    - Plans the full A* path, converts to actions, then executes sequentially from queue.
    - FORWARD steps are blind for faster operation (no expected-cell verification).
    - Before each action, checks for collision and goal change (gesture).
    - Replans only on: goal reached, collision, gesture change.
    - Goal reached when rectified pixel distance < GOAL_PX_TOL.
    """

    def pixel_goal_reached(rc: Optional[Tuple[int,int]], goal: Tuple[int,int]) -> bool:
        try:
            return rc is not None and rc_distance_px(rc, goal) < GOAL_PX_TOL
        except Exception:
            # Fallback to rect tolerance if pixel distance unavailable
            return rc is not None and rc_within_rect_tol(rc, goal)

    def current_goal_state(ws) -> str:
        """Return current goal STATE only: 'FOOD', 'HOME', or 'NONE'."""
        if ws.gesture_type == "OPEN_PALM" and ws.gesture_cell is not None:
            return "FOOD"
        if ws.gesture_type == "CLOSED_FIST" and ws.home_cell is not None:
            return "HOME"
        return "NONE"

    # Outer loop: plan-execute until goal reached or interruption requires replan
    while True:
        ws0 = get_snapshot()
        if ws0.robot_cell is None:
            print("XXX NOOOO ROBOT LOST XXX")
            send_stop()
            return False
        
            #print("MM DISTANCE: ", rc_distance_mm(ws0.robot_cell, goal_rc))
            #print("PX DISTANCE: ", rc_distance_px(ws0.robot_cell, goal_rc))

        # Early pixel-goal check
        if pixel_goal_reached(ws0.robot_cell, goal_rc):
            print("OOO GOAL REACHED (pixel tolerance) OOO")
            send_stop()
            return True

        # Plan path from current pose
        path_xy = plan_path(ws0.robot_cell, goal_rc, goaltype)
        if not path_xy or len(path_xy) < 2:
            print("[A*] no path or trivial path — waiting")
            time.sleep(0.2)
            continue

        # Convert (x,y) to (r,c) and build actions
        path_rc: List[Tuple[int,int]] = [(y, x) for (x, y) in path_xy]

        # If any cell on the path is already within pixel goal tolerance
        # stop action generation at that cell to avoid overshooting.
        stop_idx: Optional[int] = None
        try:
            for i, rc in enumerate(path_rc):
                if rc_distance_px(rc, goal_rc) < GOAL_PX_TOL:
                    stop_idx = i
                    print(f"[GOAL] path cell within {GOAL_PX_TOL}px at idx={i} rc={rc}; will stop there")
                    break
        except Exception:
            # If pixel math unavailable fall back to full path
            stop_idx = None

        def _delta_to_card(dy: int, dx: int) -> Optional[str]:
            if dx == 1 and dy == 0:
                return "E"
            if dx == -1 and dy == 0:
                return "W"
            if dy == 1 and dx == 0:
                return "S"
            if dy == -1 and dx == 0:
                return "N"
            return None

        actions: List[Tuple[str, object]] = []
        heading_card = ws0.robot_heading_card or "N"

        # Determine how many transitions to enqueue up to stop_idx
        max_transitions = (stop_idx if stop_idx is not None else (len(path_rc) - 1))
        for i in range(max_transitions):
            (y0, x0) = path_rc[i]
            (y1, x1) = path_rc[i + 1]
            dy, dx = (y1 - y0), (x1 - x0)
            tgt_card = _delta_to_card(dy, dx)
            if tgt_card is None:
                print(f"[PLAN] Non-cardinal step encountered ({dy},{dx}); replanning")
                actions = []
                break
            if heading_card != tgt_card:
                actions.append(("TURN_TO", tgt_card))
                heading_card = tgt_card
            actions.append(("FORWARD", 1))

        if not actions:
            # Replan next loop if we couldn't create actions
            time.sleep(0.05)
            continue

        # Execute queued actions in chunks with collision/goal-proximity/goal-change checks
        executed = 0
        for kind, val in actions:

            print("execute queued actions")

            # Interruption checks
            if is_collision():
                print("XXX COLLISION STOP XXX")
                send_stop()
                backup_small()
                # Replan due to collision
                break

            ws = get_snapshot()
            if ws.robot_cell is None:
                print("XXX NOOOO ROBOT LOST XXX")
                send_stop()
                return False
            
            #print("2MM DISTANCE: ", rc_distance_mm(ws.robot_cell, goal_rc))
            #print("2PX DISTANCE: ", rc_distance_px(ws.robot_cell, goal_rc))

            # Pixel-based goal tolerance
            if pixel_goal_reached(ws.robot_cell, goal_rc):
                print("OOO GOAL REACHED (pixel tolerance) OOO")
                send_stop()
                return True

            # Gesture-based goal change: react ONLY to state transitions (FOOD/HOME/NONE) (not cell changes)
            maybe_state = current_goal_state(ws)
            if maybe_state == "NONE":
                print("[GOAL] gesture lost (NONE); pausing/replanning")
                send_stop()
                return False
            if maybe_state != goaltype:
                # Update goal only when the STATE changes; ignore per-frame cell drift
                print(f"[GOAL] state changed {goaltype} -> {maybe_state}; replanning")
                goaltype = maybe_state
                if goaltype == "FOOD" and ws.gesture_cell is not None:
                    goal_rc = ws.gesture_cell
                elif goaltype == "HOME" and ws.home_cell is not None:
                    goal_rc = ws.home_cell
                # Replan due to state change
                break

            if kind == "TURN_TO":
                target_card = str(val)
                target_deg = card_to_target_deg(target_card)
                print(f"[INTENT] TURN_TO {target_card} ({target_deg:.1f} deg)")
                ok = turn_to_heading_exact(target_deg)
                if not ok:
                    print("[TURN] alignment failed; replanning")
                    break
                executed += 1
                if executed >= EXEC_CHUNK_ACTIONS:
                    print(f"[CHUNK] executed {executed} actions; replanning next chunk")
                    break
                continue

            if kind == "FORWARD":
                lspd = MOVE_SPEED * FORWARD_L_SIGN
                rspd = MOVE_SPEED * FORWARD_R_SIGN
                print(f"[INTENT] FORWARD blind ticks={CELL_FORWARD_TICKS} lspd={lspd} rspd={rspd}")

                send_ticks(CELL_FORWARD_TICKS, CELL_FORWARD_TICKS, lspd, rspd)
                ok = wait_done_or_abort(STEP_TIMEOUT, goal_rc)
                if not ok:
                    print("[MOVE] aborted or timeout; replanning")
                    break
                executed += 1
                if executed >= EXEC_CHUNK_ACTIONS:
                    print(f"[CHUNK] executed {executed} actions; replanning next chunk")
                    break
                continue

            # Unknown action kind
            print(f"[PLAN] Unknown action {kind}; replanning")
            break

        # Loop back to replan (either finished actions or interrupted)
        time.sleep(0.02)


def main():
    start_redbot()
    start_collision_monitor()

    # INA260 optional
    try:
        startINA260()
    except Exception as e:
        print(f"[INA260] not started: {e}")

    #  perception (ROS2 + hand + overlay + grid)
    start_perception(
        OVERLAY_JSON,
        GRID_JSON,
        robot_pos_id=ROBOT_POS_ID,
        robot_head_id=ROBOT_HEAD_ID,
        ros_topic=os.getenv("ROS_TOPIC", "robotPositions"),
        ros_msg_type=os.getenv("ROS_MSG_TYPE", "string"),
        min_certainty=float(os.getenv("MIN_CERTAINTY", "0.25")),
        max_speed=float(os.getenv
        ("MAX_SPEED", "500.0")),
        hand_stream_url=os.getenv("HAND_STREAM_URL", None),
        hand_show_window=os.getenv("HAND_SHOW_WINDOW", "0") == "1",
        hand_height_mm=float(os.getenv("HAND_HEIGHT_MM", "1000.0")), # default 1000mm
    )

    try:
        # Cache home (if any)
        home = get_home_cell()
        if home is None:
            print("[WARN] No HOME cell found in grid (value==2). CLOSED_FIST will do nothing.")

        while True:
            # 1) Wait for stable robot cell
            robot_rc = wait_stable_robot_cell()

            # 2) Wait for stable goal
            goaltype, goal_rc = wait_goal_cell()

            # If CLOSED_FIST but no home, idle
            if goaltype == "HOME" and (goal_rc is None):
                time.sleep(0.2)
                continue

            #print(f"[GOAL] {goaltype} goal_rc={goal_rc} start_rc={robot_rc}")
            #print("MM DISTANCE: ", rc_distance_mm(robot_rc, goal_rc))
            #print("PX DISTANCE: ", rc_distance_px(robot_rc, goal_rc))

            # 3) Plan and do
            path = plan_path(robot_rc, goal_rc, goaltype)
            if path is None:
                print("[A*] no path found — waiting and trying again.")
                time.sleep(0.5)
                continue
            print("[MODE] execute_all (prequeued)")
            reached = execute_all(goal_rc, goaltype)
            if reached:
                print("[DONE] reached goal.")
                send_stop()
                time.sleep(0.5)
            else:
                print("[REPLAN] aborted step or drift/collision — replanning.")
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExiting (Ctrl+C)")
    finally:
        try:
            send_stop()
        except Exception:
            pass
        try:
            stop_perception()
        except Exception:
            pass
        try:
            stopINA260()
        except Exception:
            pass
        try:
            collision_cleanup()
        except Exception:
            pass
        try:
            stop_redbot()
        except Exception:
            pass


if __name__ == "__main__":
    main()
