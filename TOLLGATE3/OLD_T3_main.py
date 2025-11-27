#!/usr/bin/env python3
"""
T3_main.py - Standalone RedBot Navigation with Hand Gesture Control

This script orchestrates the RedBot to navigate using hand gesture input.
- Detects hand gestures (Open_Palm → go to hand, Closed_Fist → go home)
- Converts hand position to grid cell using overlay API
- Navigates robot to the goal cell
- Logs power consumption via INA260

Fully standalone: searches for local APIs, no external Integration-main imports.
"""

import ina260
import redbot
from ina260 import getCurrent, getVoltage, stopINA260
import time
import sys
import os
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


# ============================================================================
# Utility Functions: Module Loading & Config Discovery
# ============================================================================

def _load_module_from_path(path, modname):
    """Load a Python module from a file path. Returns None if not found/error."""
    path = str(path)
    if not os.path.isfile(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"[WARN] Failed to load module {path}: {e}", flush=True)
        return None

def _find_file_in_candidates(candidate_paths, filename_hint=""):
    """Find first existing file from a list of candidate paths."""
    for p in candidate_paths:
        if os.path.isfile(str(p)):
            return p
    return None

def printINA260():
    """Read and print current INA260 power metrics."""
    print("Reading INA260...", flush=True)
    try:
        I = getCurrent()
        V = getVoltage()
        print(f"V = {V:.3f} V, I = {I:.3f} A", flush=True)
    except Exception as e:
        print(f"Error reading INA260: {e}", flush=True)


def main():
    """Main orchestrator: initialize APIs, detect gesture, navigate robot."""
    # Start background threads for INA260 and RedBot
    ina260.startINA260(bus_num=1, address=0x40, interval=0.2)
    redbot.start_redbot()

    base_dir = os.path.dirname(__file__)
    integration_root = os.path.join(base_dir, "Integration-main")

    # ========================================================================
    # Setup sys.path to find real API implementations
    # (Using sys.path injection like integ-main.py does)
    # ========================================================================
    if integration_root and os.path.isdir(integration_root):
        # Add API paths in priority order (prefer real implementations)
        sys.path.insert(0, os.path.join(integration_root, "apis", "overlay-api"))
        sys.path.insert(0, os.path.join(integration_root, "apis", "layout-api"))
        sys.path.insert(0, os.path.join(integration_root, "apis", "hand-recognition-api"))
        sys.path.insert(0, os.path.join(integration_root, "apis", "ros2-api"))
        sys.path.insert(0, integration_root)

    # ========================================================================
    # Load Real API Implementations (prefer real over shims)
    # ========================================================================

    # 1. ROS2 Position API
    RobotPositionAPI = None
    try:
        ros2_path = os.path.join(integration_root, "apis", "ros2-api", "ros2-api.py")
        if os.path.isfile(ros2_path):
            import runpy
            ns = runpy.run_path(ros2_path)
            if ns and 'RobotPositionAPI' in ns:
                RobotPositionAPI = ns['RobotPositionAPI']
                print(f"[OK] Loaded RobotPositionAPI from {ros2_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load RobotPositionAPI: {e}", flush=True)

    # 2. Overlay API (prefer real implementation in overlay/ directory)
    GPSOverlay = None
    overlay_candidates = [
        os.path.join(integration_root, "overlay", "overlay-api.py"),  # Real implementation
        os.path.join(integration_root, "apis", "overlay-api", "overlay-api.py"),  # Shim
    ]
    for p in overlay_candidates:
        if os.path.isfile(p):
            try:
                mod = _load_module_from_path(p, f"overlay_api_{id(p)}")
                if mod and hasattr(mod, "GPSOverlay"):
                    GPSOverlay = getattr(mod, "GPSOverlay")
                    print(f"[OK] Loaded GPSOverlay from {p}", flush=True)
                    break
            except Exception as e:
                print(f"[WARN] Failed to load from {p}: {e}", flush=True)

    # 3. Layout API (prefer real implementation in object-layout/)
    get_map = None
    layout_candidates = [
        os.path.join(integration_root, "object-layout", "api", "layout-api.py"),  # Real
        os.path.join(integration_root, "apis", "layout-api", "layout-api.py"),  # Shim
    ]
    for p in layout_candidates:
        if os.path.isfile(p):
            try:
                mod = _load_module_from_path(p, f"layout_api_{id(p)}")
                if mod and hasattr(mod, "get_map"):
                    get_map = getattr(mod, "get_map")
                    print(f"[OK] Loaded get_map from {p}", flush=True)
                    break
            except Exception as e:
                print(f"[WARN] Failed to load from {p}: {e}", flush=True)

    # Find gps_overlay.json for overlay initialization
    gps_overlay_candidates = [
        os.path.join(base_dir, "Integration-main", "apis", "overlay-api", "gps_overlay.json"),
        os.path.join(base_dir, "Integration-main", "overlay", "gps_overlay.json"),
        os.path.join(base_dir, "apis", "overlay-api", "gps_overlay.json"),
    ]
    gps_overlay_path = _find_file_in_candidates(gps_overlay_candidates)
    if gps_overlay_path:
        print(f"[OK] Found gps_overlay.json at {gps_overlay_path}", flush=True)
    else:
        print("[WARN] gps_overlay.json not found", flush=True)

    # No internal ROS2 fallbacks; rely on Integration ros2-api

    class _StubGPSOverlay:
        """Stub: returns dummy grid cell."""
        def __init__(self, config_path=None):
            pass

        def get_grid_cell(self, x, y):
            return {"col": 0, "row": 0, "in_bounds": False}

        def get_grid_cell_from_rectified(self, x, y):
            return {"col": 0, "row": 0, "in_bounds": False}

    def _get_map_stub():
        """Stub: returns empty 8x8 grid."""
        return [[0 for _ in range(8)] for _ in range(8)]

    # ========================================================================
    # Instantiate APIs
    # ========================================================================

    # Create API instances
    if RobotPositionAPI:
        # Allow overriding ROS2 settings via environment variables
        topic = os.environ.get("T3_ROS2_TOPIC", "robotPositions")
        msg_type = os.environ.get("T3_ROS2_MSG_TYPE", "string")  # 'string' or 'float32multiarray'
        try:
            min_cert_env = os.environ.get("T3_MIN_CERTAINTY")
            min_cert_val = float(min_cert_env) if min_cert_env is not None else 0.25
        except Exception:
            min_cert_val = 0.25
        try:
            tick_hz_env = os.environ.get("T3_ROS2_TICK_HZ")
            tick_hz_val = float(tick_hz_env) if tick_hz_env is not None else 5.0
        except Exception:
            tick_hz_val = 5.0
        # QoS overrides
        reliability = os.environ.get("T3_ROS2_RELIABILITY", "reliable")  # reliable|best_effort
        durability = os.environ.get("T3_ROS2_DURABILITY", "volatile")    # volatile|transient_local
        try:
            depth_val = int(os.environ.get("T3_ROS2_QOS_DEPTH", "10"))
        except Exception:
            depth_val = 10

        # normalize to absolute topic
        if not str(topic).startswith('/'):
            topic = '/' + str(topic)
        try:
            ros_api = RobotPositionAPI(topic=topic, msg_type=msg_type,
                                        min_certainty=min_cert_val, tick_hz=tick_hz_val,
                                        reliability=reliability, durability=durability, depth=depth_val)
            print(f"[INFO] ROS2 config: topic='{topic}', msg_type='{msg_type}', min_certainty={min_cert_val}, tick_hz={tick_hz_val}, qos(rel={reliability}, dur={durability}, depth={depth_val})", flush=True)
        except Exception as e:
            print(f"[ERROR] RobotPositionAPI ctor failed: {e}", flush=True)
            stopINA260()
            redbot.stop_redbot()
            sys.exit(1)
    else:
        print("[ERROR] RobotPositionAPI not available; ensure Integration-main is present", flush=True)
        stopINA260()
        redbot.stop_redbot()
        sys.exit(1)

    if GPSOverlay:
        try:
            overlay = GPSOverlay(gps_overlay_path) if gps_overlay_path else GPSOverlay()
        except Exception as e:
            print(f"[WARN] Failed to init GPSOverlay: {e}, using stub", flush=True)
            overlay = _StubGPSOverlay()
    else:
        overlay = _StubGPSOverlay()

    map_func = get_map if get_map else _get_map_stub

    # Start ROS2 API if available
    try:
        if hasattr(ros_api, 'start') and callable(ros_api.start):
            ros_api.start()
            print("[OK] RobotPositionAPI started", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to start RobotPositionAPI: {e}", flush=True)

    # ========================================================================
    # Load Navigator
    # ========================================================================

    nav_candidates = [
        os.path.join(base_dir, "Integration-main", "navigation.py"),
        os.path.join(base_dir, "navigation.py"),
    ]
    Navigator = None
    for p in nav_candidates:
        nav_mod = _load_module_from_path(p, "navigation_mod")
        if nav_mod and hasattr(nav_mod, "Navigator"):
            Navigator = getattr(nav_mod, "Navigator")
            print(f"[OK] Loaded Navigator from {p}", flush=True)
            break

    if Navigator is None:
        print("[ERROR] Navigator not found, cannot continue", flush=True)
        stopINA260()
        redbot.stop_redbot()
        sys.exit(1)

    # Instantiate Navigator
    nav = Navigator(
        ros_api=ros_api,
        overlay=overlay,
        get_map_func=map_func,
        ticks_per_cell=192,
        turn_ticks=96,
        forward_speed=50,
        turn_speed=25,
        cmd_timeout=8.0,
        manage_redbot=False,  # main started redbot, navigator should not stop it
    )
    nav.start()
    print("[OK] Navigator started", flush=True)

    try:
        # Initial power reading
        printINA260()

        # Test short move
        print("[INFO] Short test move (forward 192 ticks)...", flush=True)
        ok = redbot.send_ticks_blocking(192, 192, 50, 50, timeout=5.0)
        print(f"[INFO] Test move result: {ok}", flush=True)

        # ====================================================================
        # Step 1: Get start position from ROS2 Position API (spiral 0)
        # ====================================================================
        print("[INFO] Waiting for start position (spiral=0) from ROS2 API...", flush=True)
        start_row = None
        # Allow target spiral id & certainty override
        try:
            target_spiral_id = int(os.environ.get("T3_SPIRAL_ID", "0"))
        except Exception:
            target_spiral_id = 0
        # New: allow list of candidate ids (comma separated) e.g. "0,1"; if provided overrides single target id logic
        raw_ids = os.environ.get("T3_SPIRAL_IDS")
        candidate_ids = []
        if raw_ids:
            try:
                candidate_ids = [int(x.strip()) for x in raw_ids.split(',') if x.strip() != '']
            except Exception:
                candidate_ids = []
        # Selection mode: first (default), best (highest certainty), stable (same id N times)
        select_mode = os.environ.get("T3_SPIRAL_SELECT_MODE", "first").strip().lower()
        try:
            stable_needed = int(os.environ.get("T3_SPIRAL_STABLE_COUNT", "3"))
        except Exception:
            stable_needed = 3
        stable_tracker_id = None
        stable_tracker_count = 0
        try:
            min_certainty = float(os.environ.get("T3_START_MIN_CERT", "0.5"))
        except Exception:
            min_certainty = 0.5
        base_min_certainty = min_certainty
        # Adaptive lowering floor so we can still acquire start row when publisher certainty is low
        adaptive_floor = float(os.environ.get("T3_START_MIN_CERT_FLOOR", "0.30"))
        wait_timeout = 60.0
        t0 = time.time()
        last_log = 0.0
        while time.time() - t0 < wait_timeout:
            try:
                rows = ros_api.getPosition()  # list of SpiralRow
                if rows:
                    # Dynamically lower min_certainty over time (10s, 20s, 30s) if nothing found
                    elapsed = time.time() - t0
                    if not start_row:
                        # Compute adaptive threshold steps: each 10s reduce by 0.05 until floor
                        reductions = int(elapsed // 10)
                        dynamic_min = max(adaptive_floor, base_min_certainty - 0.05 * reductions)
                    else:
                        dynamic_min = base_min_certainty
                    iterable_rows = (rows if isinstance(rows, list) else [rows])
                    # Candidate selection logic
                    selected = None
                    if candidate_ids:
                        # Filter by candidate ids and certainty threshold
                        eligible = [r for r in iterable_rows if getattr(r, 'id', None) in candidate_ids and getattr(r, 'certainty', 1.0) >= dynamic_min]
                        if eligible:
                            if select_mode == 'best':
                                selected = max(eligible, key=lambda r: getattr(r, 'certainty', 0.0))
                            elif select_mode == 'stable':
                                # Pick highest certainty, then require same id stable_needed consecutive cycles
                                top = max(eligible, key=lambda r: getattr(r, 'certainty', 0.0))
                                tid = getattr(top, 'id', None)
                                if stable_tracker_id is None or tid != stable_tracker_id:
                                    stable_tracker_id = tid
                                    stable_tracker_count = 1
                                else:
                                    stable_tracker_count += 1
                                if stable_tracker_count >= stable_needed:
                                    selected = top
                            else:  # first
                                selected = eligible[0]
                    else:
                        # Single target id logic
                        for r in iterable_rows:
                            rid = getattr(r, 'id', None)
                            cert = getattr(r, 'certainty', 1.0)
                            if rid == target_spiral_id and cert >= dynamic_min:
                                selected = r
                                break
                    if selected is not None:
                        start_row = selected
                        # Log selection details
                        print(f"[ROS2] Selected start id={getattr(selected,'id',None)} cert={getattr(selected,'certainty',None):.2f} dynamic_min={dynamic_min:.2f} mode={select_mode} candidates={candidate_ids if candidate_ids else [target_spiral_id]}", flush=True)
                        break
                    if start_row:
                        break
                    # Diagnostic: log what we are receiving once per second
                    now = time.time()
                    if now - last_log > 1.0:
                        summary = ", ".join(
                            [f"id={getattr(r,'id',None)} r={getattr(r,'row',None):.2f} c={getattr(r,'col',None):.2f} cert={getattr(r,'certainty',None):.2f}"
                             for r in (rows if isinstance(rows, list) else [rows])][:5]
                        )
                        print(f"[ROS2] Received rows (target={target_spiral_id} candidates={candidate_ids or 'N/A'} mode={select_mode} base_min={base_min_certainty:.2f} dyn_min={dynamic_min:.2f} floor={adaptive_floor:.2f} stable={stable_tracker_id}:{stable_tracker_count}/{stable_needed}): {summary}", flush=True)
                        last_log = now
            except Exception:
                pass
            time.sleep(0.25)

        if start_row is None:
            print(f"[ERROR] Start position for spiral={target_spiral_id} not found within timeout; aborting.", flush=True)
            raise RuntimeError("Start position not found")
        else:
            start_cell = (int(round(start_row.row)), int(round(start_row.col)))
            heading = getattr(start_row, 'angle', None)
            print(f"[OK] Start cell (spiral={target_spiral_id}): {start_cell}, angle={heading}", flush=True)

        # ====================================================================
        # Step 2: Small forward move to stabilize heading
        # ====================================================================
        print("[INFO] Short forward move to stabilize heading...", flush=True)
        ok = redbot.send_ticks_blocking(96, 96, 120, 120, timeout=5.0)
        print(f"[INFO] Stabilization move result: {ok}", flush=True)

        # Read position again
        try:
            rows2 = ros_api.getPosition()
            if rows2:
                r = rows2[0] if isinstance(rows2, list) else rows2
                heading = getattr(r, 'angle', None)
                print(f"[INFO] Heading after move: {heading}", flush=True)
        except Exception:
            pass

        # ====================================================================
        # Step 3: Detect hand gesture and map to grid cell
        # ====================================================================
        print("[INFO] Attempting to detect hand gesture...", flush=True)
        hand_cell = None
        gesture_label = None

        # Try to load hand recognition API (prefer real implementation in hand_recognition/)
        hand_candidates = [
            os.path.join(integration_root, "hand_recognition", "hand-recognition-api.py"),  # Real
            os.path.join(integration_root, "apis", "hand-recognition-api", "hand-recognition-api.py"),  # Shim
        ]

        gr = None
        for p in hand_candidates:
            if os.path.isfile(p):
                try:
                    mod = _load_module_from_path(p, f"hand_recognition_api_{id(p)}")
                    if mod and hasattr(mod, "GestureRecognizer"):
                        GestureRecognizer = getattr(mod, "GestureRecognizer")
                        gr = GestureRecognizer()
                        print(f"[OK] Loaded GestureRecognizer from {p}", flush=True)
                        break
                except Exception as e:
                    print(f"[WARN] Failed to load GestureRecognizer from {p}: {e}", flush=True)

        if gr:
            # Start recognizer (prefer non-blocking run(), fallback to start())
            try:
                if hasattr(gr, 'run') and callable(gr.run):
                    gr.run()
                    print("[OK] GestureRecognizer.run() started", flush=True)
                elif hasattr(gr, 'start') and callable(gr.start):
                    gr.start()
                    print("[OK] GestureRecognizer.start() started", flush=True)
            except Exception as e:
                print(f"[WARN] Failed to start GestureRecognizer: {e}", flush=True)

            # Warm up
            time.sleep(0.5)

            # Read gesture and position
            try:
                pos_x = pos_y = None
                if hasattr(gr, 'get_position') and callable(gr.get_position):
                    pos = gr.get_position()
                    if pos and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        if pos[0] is not None and pos[1] is not None:
                            pos_x, pos_y = float(pos[0]), float(pos[1])

                if hasattr(gr, 'get_gesture') and callable(gr.get_gesture):
                    gesture_label = gr.get_gesture()
                else:
                    gesture_label = None

                print(f"[INFO] Gesture: {gesture_label}, Hand pos: ({pos_x}, {pos_y})", flush=True)

                # Map hand position to grid cell
                if pos_x is not None and pos_y is not None:
                    try:
                        if hasattr(overlay, 'get_grid_cell_from_rectified'):
                            cell = overlay.get_grid_cell_from_rectified(pos_x, pos_y)
                        elif hasattr(overlay, 'get_grid_cell'):
                            cell = overlay.get_grid_cell(pos_x, pos_y)
                        else:
                            cell = None

                        if cell and cell.get("in_bounds"):
                            hand_cell = (int(cell["row"]), int(cell["col"]))
                            print(f"[OK] Hand mapped to cell {hand_cell}", flush=True)
                    except Exception as e:
                        print(f"[WARN] Failed to map hand position to cell: {e}", flush=True)
            except Exception as e:
                print(f"[WARN] Failed to read gesture/position: {e}", flush=True)

            # Stop recognizer
            try:
                if hasattr(gr, 'stop') and callable(gr.stop):
                    gr.stop()
                    print("[OK] GestureRecognizer stopped", flush=True)
            except Exception as e:
                print(f"[WARN] Failed to stop GestureRecognizer: {e}", flush=True)
        else:
            print("[WARN] GestureRecognizer not available; skipping gesture navigation", flush=True)

        # ====================================================================
        # Step 4: Decide goal based on gesture
        # ====================================================================
        goal_cell = None

        if gesture_label and gesture_label.upper().startswith("FIST"):
            # Go home: find HOME cell or use (0, 0)
            print("[INFO] Gesture = FIST → navigating to home", flush=True)
            try:
                grid = map_func()
                for r_i, row in enumerate(grid):
                    for c_i, v in enumerate(row):
                        if v > 1:  # non-zero, non-free cell
                            goal_cell = (r_i, c_i)
                            raise StopIteration
            except StopIteration:
                pass
            except Exception as e:
                print(f"[WARN] Error scanning grid: {e}", flush=True)
            if goal_cell is None:
                goal_cell = (0, 0)
                print(f"[INFO] No HOME found in grid, using default (0,0)", flush=True)

        elif gesture_label and gesture_label.upper().startswith("PALM") and hand_cell:
            # Go to hand
            goal_cell = hand_cell
            print(f"[INFO] Gesture = PALM → navigating to hand at {hand_cell}", flush=True)

        else:
            print("[WARN] No valid gesture/hand cell detected → no navigation", flush=True)

        # ====================================================================
        # Step 5: Navigate to goal
        # ====================================================================
        if goal_cell:
            print(f"[INFO] Navigating from {start_cell} to goal {goal_cell}...", flush=True)
            try:
                nav.goto(goal_cell, wait=True)
                print(f"[OK] Navigation to {goal_cell} complete", flush=True)
            except Exception as e:
                print(f"[ERROR] Navigation failed: {e}", flush=True)
        else:
            print("[INFO] No goal to navigate to", flush=True)

        # Final power reading
        printINA260()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, shutting down...", flush=True)

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        # ====================================================================
        # Cleanup: stop all subsystems gracefully
        # ====================================================================
        print("[INFO] Cleaning up...", flush=True)

        # Stop INA260 reader
        try:
            stopINA260()
            print("[OK] INA260 stopped", flush=True)
        except Exception as e:
            print(f"[WARN] Error stopping INA260: {e}", flush=True)

        # Stop ROS2 API
        try:
            if hasattr(ros_api, 'stop') and callable(ros_api.stop):
                ros_api.stop()
                print("[OK] RobotPositionAPI stopped", flush=True)
        except Exception as e:
            print(f"[WARN] Error stopping RobotPositionAPI: {e}", flush=True)

        # Stop Navigator (does not stop redbot since manage_redbot=False)
        try:
            nav.stop()
            print("[OK] Navigator stopped", flush=True)
        except Exception as e:
            print(f"[WARN] Error stopping Navigator: {e}", flush=True)

        # Stop RedBot
        try:
            redbot.stop_redbot()
            print("[OK] RedBot stopped", flush=True)
        except Exception as e:
            print(f"[WARN] Error stopping RedBot: {e}", flush=True)

        print("[OK] Shutdown complete", flush=True)


if __name__ == "__main__":
    main()