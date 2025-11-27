#!/usr/bin/env python3
"""
T3_main.py — Minimal run using Mermaid-Integration APIs only

Flow:
1) Init INA260 + RedBot; drive a little forward; print INA260.
2) Start ROS2 RobotPositionAPI; wait until start position (ids in [0,1]) with min certainty.
3) Drive a bit forward to stabilize; read angle once for logging.
4) Start HandRecognizer; wait for PALM or FIST; PALM → small forward, FIST → small backward.
5) Clean shutdown.
"""

import os
import sys
import time
import types
import importlib.util
from typing import List
from dataclasses import dataclass

import ina260
import redbot
from ina260 import getCurrent, getVoltage, stopINA260


def _load_module_from_path(path: str, modname: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure module is visible in sys.modules during execution (needed for dataclasses type resolution)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def _ensure_abs_topic(topic: str) -> str:
    topic = topic.strip()
    return topic if topic.startswith('/') else '/' + topic


def _env_list_int(var: str, default: str) -> List[int]:
    raw = os.environ.get(var, default)
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


def _print_ina260():
    print("Reading INA260...", flush=True)
    try:
        V, I = getVoltage(), getCurrent()
        print(f"V = {V:.3f} V, I = {I:.3f} A", flush=True)
    except Exception as e:
        print(f"[WARN] INA260 read failed: {e}", flush=True)


def main():
    base_dir = os.path.dirname(__file__)
    mroot = os.path.join(base_dir, 'Mermaid-Integration')

    # Resolve API paths
    ros2_api_py = os.path.join(mroot, 'apis', 'ros2-api', 'ros2-api.py')
    overlay_py = os.path.join(mroot, 'apis', 'overlay-api', 'overlay.py')
    layout_api_py = os.path.join(mroot, 'apis', 'layout-api', 'layout-api.py')
    hand_rec_py = os.path.join(mroot, 'apis', 'hand-recognition-api', 'hand_recognition.py')

    # Provide alias module required by Mermaid ROS2 API: `from ros2 import SpiralRow`
    @dataclass
    class SpiralRow:
        id: int
        row: float
        col: float
        angle: float
        certainty: float

    ros2_alias = types.ModuleType('ros2')
    ros2_alias.SpiralRow = SpiralRow
    sys.modules.setdefault('ros2', ros2_alias)

    # Load Mermaid APIs directly (after injecting alias)
    ros2_mod = _load_module_from_path(ros2_api_py, 'mermaid_ros2_api')
    overlay_mod = _load_module_from_path(overlay_py, 'mermaid_overlay')
    layout_mod = _load_module_from_path(layout_api_py, 'mermaid_layout')

    # Alias expected by hand_recognition.py: api.modules.gps_overlay.GPSOverlay
    api_pkg = types.ModuleType('api')
    modules_pkg = types.ModuleType('api.modules')
    gps_overlay_alias = types.ModuleType('api.modules.gps_overlay')
    sys.modules.setdefault('api', api_pkg)
    sys.modules.setdefault('api.modules', modules_pkg)
    sys.modules.setdefault('api.modules.gps_overlay', gps_overlay_alias)

    # Resolve GPSOverlay now and export it into the alias BEFORE loading hand module
    GPSOverlay = getattr(overlay_mod, 'GPSOverlay')
    sys.modules['api.modules.gps_overlay'].GPSOverlay = GPSOverlay

    # Now it is safe to load the hand recognition module
    hand_mod = _load_module_from_path(hand_rec_py, 'mermaid_hand_rec')

    RobotPositionAPI = getattr(ros2_mod, 'RobotPositionAPI')
    get_map = getattr(layout_mod, 'get_map')  # kept if needed later
    HandRecognizer = getattr(hand_mod, 'HandRecognizer')

    # Config
    topic = _ensure_abs_topic(os.environ.get('T3_ROS2_TOPIC', '/robotPositions'))
    msg_type = os.environ.get('T3_ROS2_MSG_TYPE', 'string')
    tick_hz = float(os.environ.get('T3_ROS2_TICK_HZ', '5.0'))
    candidate_ids = _env_list_int('T3_SPIRAL_IDS', '0,1')
    min_cert = float(os.environ.get('T3_START_MIN_CERT', '0.45'))
    wait_timeout = float(os.environ.get('T3_WAIT_TIMEOUT', '60'))
    gesture_timeout = float(os.environ.get('T3_GESTURE_TIMEOUT', '20'))

    fwd_ticks = int(os.environ.get('T3_FWD_TICKS', '96'))
    back_ticks = int(os.environ.get('T3_BACK_TICKS', '64'))
    fwd_speed = int(os.environ.get('T3_FWD_SPEED', '100'))
    back_speed = int(os.environ.get('T3_BACK_SPEED', '100'))

    # Init INA260 + RedBot
    ina260.startINA260(bus_num=1, address=0x40, interval=0.2)
    redbot.start_redbot()
    _print_ina260()

    print(f"[INFO] Sanity forward move: {fwd_ticks} ticks @ {fwd_speed}", flush=True)
    ok = redbot.send_ticks_blocking(fwd_ticks, fwd_ticks, fwd_speed, fwd_speed, timeout=6.0)
    print(f"[OK] Sanity move result: {ok}", flush=True)

    # Start ROS2 API and wait for start position
    ros_api = RobotPositionAPI(topic=topic, msg_type=msg_type, min_certainty=0.0,
                               max_speed=500.0, tick_hz=tick_hz)
    ros_api.start()
    print(f"[OK] RobotPositionAPI started topic='{topic}' type='{msg_type}'", flush=True)

    print(f"[INFO] Waiting for start position ids={candidate_ids} min_cert={min_cert:.2f} ...", flush=True)
    t0 = time.time()
    start_row = None
    while time.time() - t0 < wait_timeout:
        rows = ros_api.getPosition()
        if rows:
            eligible = [r for r in rows if getattr(r, 'id', None) in candidate_ids and getattr(r, 'certainty', 0.0) >= min_cert]
            if eligible:
                start_row = max(eligible, key=lambda r: getattr(r, 'certainty', 0.0))
                print(f"[OK] Start id={start_row.id} row={start_row.row:.2f} col={start_row.col:.2f} ang={getattr(start_row,'angle',None)} cert={start_row.certainty:.2f}", flush=True)
                break
        time.sleep(0.2)

    if start_row is None:
        print("[ERROR] Start position not found within timeout; exiting.", flush=True)
        raise SystemExit(2)

    # Stabilize heading
    print("[INFO] Stabilize heading: short forward move", flush=True)
    ok = redbot.send_ticks_blocking(max(32, fwd_ticks//2), max(32, fwd_ticks//2), fwd_speed, fwd_speed, timeout=5.0)
    print(f"[OK] Stabilization move result: {ok}", flush=True)
    try:
        rows2 = ros_api.getPosition()
        if rows2:
            best = None
            for r in rows2:
                if r.id == start_row.id:
                    best = r
                    break
            ang = getattr(best or rows2[0], 'angle', None)
            print(f"[INFO] Heading (angle): {ang}", flush=True)
    except Exception:
        pass

    # Gesture
    print("[INFO] Waiting for hand gesture (PALM/FIST)...", flush=True)
    # Configure hand recognizer: smaller window, optional display
    show_window = os.environ.get('T3_HAND_SHOW_WINDOW', '0') == '1'
    window_scale = float(os.environ.get('T3_HAND_WINDOW_SCALE', '0.5'))
    min_det_conf = float(os.environ.get('T3_HAND_MIN_DET', '0.5'))
    min_track_conf = float(os.environ.get('T3_HAND_MIN_TRACK', '0.5'))
    gr = HandRecognizer(show_window=show_window, window_scale=window_scale,
                        min_det_conf=min_det_conf, min_track_conf=min_track_conf)
    if hasattr(gr, 'run') and callable(gr.run):
        gr.run()
    elif hasattr(gr, 'start') and callable(gr.start):
        gr.start()

    gesture = None
    pos_x = pos_y = None
    gt0 = time.time()
    try:
        while time.time() - gt0 < gesture_timeout:
            if hasattr(gr, 'get_gesture') and callable(gr.get_gesture):
                gesture = gr.get_gesture()
            if hasattr(gr, 'get_position') and callable(gr.get_position):
                p = gr.get_position()
                if p and isinstance(p, (list, tuple)) and len(p) >= 2:
                    if p[0] is not None and p[1] is not None:
                        try:
                            pos_x, pos_y = float(p[0]), float(p[1])
                        except Exception:
                            pos_x = pos_y = None
            if gesture:
                print(f"[INFO] Gesture={gesture} pos=({pos_x}, {pos_y})", flush=True)
                g = str(gesture).upper()
                if g.startswith('FIST'):
                    print(f"[ACT] FIST → backward {back_ticks} ticks", flush=True)
                    redbot.send_ticks_blocking(-back_ticks, -back_ticks, back_speed, back_speed, timeout=5.0)
                    break
                if g.startswith('PALM'):
                    print(f"[ACT] PALM → forward {fwd_ticks} ticks", flush=True)
                    redbot.send_ticks_blocking(fwd_ticks, fwd_ticks, fwd_speed, fwd_speed, timeout=5.0)
                    break
            time.sleep(0.1)
    finally:
        if hasattr(gr, 'stop') and callable(gr.stop):
            try:
                gr.stop()
            except Exception:
                pass
        print("[OK] GestureRecognizer stopped", flush=True)

    # Final INA260
    _print_ina260()

    # Cleanup
    try:
        ros_api.stop()
        print("[OK] RobotPositionAPI stopped", flush=True)
    except Exception:
        pass
    try:
        stopINA260()
        print("[OK] INA260 stopped", flush=True)
    except Exception:
        pass
    try:
        redbot.stop_redbot()
        print("[OK] RedBot stopped", flush=True)
    except Exception:
        pass


if __name__ == '__main__':
    main()