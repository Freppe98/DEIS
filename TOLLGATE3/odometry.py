#!/usr/bin/env python3
"""
odometry.py

Differential-drive odometry for the Crab robot.

Assumptions:
- Arduino RedBot sends "DONE <leftTicks> <rightTicks>" over serial.
- redbot.py parses that and exposes get_raw_ticks(), which returns
  the *cumulative signed* encoder ticks (curL, curR), or None if not yet known.

This module:
- Runs a background thread that:
  - Polls get_raw_ticks() at some Hz
  - Computes signed tick deltas since last update
  - Integrates pose (x, y, theta) and velocities
- Provides:

    start_odometry(wheel_radius, ticks_per_rev, wheel_base, loop_hz=50.0)
    get_odometry_snapshot()

You must:
- Call start_redbot() from redbot.py somewhere before/around start_odometry()
- Have Arduino firmware that prints "DONE L R" with cumulative ticks.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from redbot import get_raw_ticks  # make sure redbot.py defines this


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OdometryState:
    # Global pose (same frame you’ll use in Kalman/world_perception if you want)
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians

    # Instantaneous velocities (approx, from last integration step)
    v: float = 0.0      # linear velocity [m/s]
    omega: float = 0.0  # angular velocity [rad/s]

    # Raw encoder counts (cumulative, as reported by RedBot)
    ticks_left_raw: int = 0
    ticks_right_raw: int = 0

    # Timestamp of last update
    timestamp: float = field(default_factory=time.time)


_odom_state = OdometryState()
_odom_lock = threading.Lock()
_odom_thread: Optional[threading.Thread] = None

# Config (set on start)
_wheel_radius: float = 0.0325     # meters
_ticks_per_rev: int = 170
_wheel_base: float = 0.157       # meters (distance between wheels)
_loop_hz: float = 50.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _odometry_loop():
    """Background thread: integrates odometry at _loop_hz."""
    global _odom_state

    if _ticks_per_rev <= 0:
        raise RuntimeError("ticks_per_rev must be > 0")

    meters_per_tick = (2.0 * math.pi * _wheel_radius) / float(_ticks_per_rev)
    dt_nominal = 1.0 / _loop_hz

    last_ticks_left: Optional[int] = None
    last_ticks_right: Optional[int] = None
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time
        if dt <= 0.0:
            dt = dt_nominal
        last_time = now

        raw = get_raw_ticks()
        if raw is None:
            # No tick info yet; just sleep and try again
            time.sleep(dt_nominal)
            continue

        ticks_left, ticks_right = raw

        # First valid reading: initialize and skip integration this cycle
        if last_ticks_left is None or last_ticks_right is None:
            last_ticks_left = ticks_left
            last_ticks_right = ticks_right
            with _odom_lock:
                _odom_state.ticks_left_raw = ticks_left
                _odom_state.ticks_right_raw = ticks_right
                _odom_state.timestamp = now
            time.sleep(dt_nominal)
            continue

        # Signed tick deltas (since Arduino ticks are cumulative signed)
        dL_ticks = ticks_left  - last_ticks_left
        dR_ticks = ticks_right - last_ticks_right

        last_ticks_left = ticks_left
        last_ticks_right = ticks_right

        # Convert ticks -> distance [m]
        dL = dL_ticks * meters_per_tick
        dR = dR_ticks * meters_per_tick

        # Differential drive kinematics
        dS = (dR + dL) / 2.0
        dTheta = 0.0
        if _wheel_base > 0.0:
            dTheta = (dR - dL) / _wheel_base

        with _odom_lock:
            x = _odom_state.x
            y = _odom_state.y
            theta = _odom_state.theta

            theta_mid = theta + dTheta / 2.0
            x_new = x + dS * math.cos(theta_mid)
            y_new = y + dS * math.sin(theta_mid)
            theta_new = _wrap_angle(theta + dTheta)

            # Simple velocity estimates
            v = 0.0
            omega = 0.0
            if dt > 0.0:
                v = dS / dt
                omega = dTheta / dt

            _odom_state.x = x_new
            _odom_state.y = y_new
            _odom_state.theta = theta_new
            _odom_state.v = v
            _odom_state.omega = omega
            _odom_state.ticks_left_raw = ticks_left
            _odom_state.ticks_right_raw = ticks_right
            _odom_state.timestamp = now

        time.sleep(dt_nominal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_odometry(
    wheel_radius: float,
    ticks_per_rev: int,
    wheel_base: float,
    loop_hz: float = 50.0,
):
    """
    Start the odometry background thread.

    Args:
        wheel_radius: wheel radius in meters
        ticks_per_rev: encoder ticks per wheel revolution
        wheel_base: distance between left and right wheels (meters)
        loop_hz: how often to integrate odometry

    Usage (e.g. in crab_main.py):

        from redbot import start_redbot
        from odometry import start_odometry, get_odometry_snapshot

        start_redbot()
        start_odometry(
            wheel_radius=0.032,   # example: 32 mm radius
            ticks_per_rev=192,    # depends on your encoder
            wheel_base=0.12,      # example: 12 cm between wheels
        )

        while True:
            odom = get_odometry_snapshot()
            print(odom.x, odom.y, odom.theta)
            time.sleep(0.1)
    """
    global _wheel_radius, _ticks_per_rev, _wheel_base, _loop_hz, _odom_thread

    _wheel_radius = float(wheel_radius)
    _ticks_per_rev = int(ticks_per_rev)
    _wheel_base = float(wheel_base)
    _loop_hz = float(loop_hz)

    if _odom_thread is not None and _odom_thread.is_alive():
        return

    t = threading.Thread(
        target=_odometry_loop,
        name="Odometry",
        daemon=True,
    )
    _odom_thread = t
    t.start()


def get_odometry_snapshot() -> OdometryState:
    """
    Return a copy of the current OdometryState.
    Safe to call from any thread.
    """
    with _odom_lock:
        return OdometryState(
            x=_odom_state.x,
            y=_odom_state.y,
            theta=_odom_state.theta,
            v=_odom_state.v,
            omega=_odom_state.omega,
            ticks_left_raw=_odom_state.ticks_left_raw,
            ticks_right_raw=_odom_state.ticks_right_raw,
            timestamp=_odom_state.timestamp,
        )


def set_odometry_pose(x: float, y: float, theta: float) -> None:
    """Synchronize odometry pose to a given (x,y,theta) in meters/radians.

    Also resets the internal tick baseline so subsequent integration proceeds
    from the current raw encoder values without causing a jump.

    Call this right after obtaining an absolute pose from world perception
    at startup, or whenever you want to re-anchor odometry.
    """
    global _odom_state
    with _odom_lock:
        _odom_state.x = float(x)
        _odom_state.y = float(y)
        _odom_state.theta = float(theta)
        # Reset last ticks by pretending current raw is the baseline
        raw = get_raw_ticks()
        if raw is not None:
            _odom_state.ticks_left_raw = int(raw[0])
            _odom_state.ticks_right_raw = int(raw[1])
        _odom_state.v = 0.0
        _odom_state.omega = 0.0
        _odom_state.timestamp = time.time()


# ---------------------------------------------------------------------------
# Test main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from redbot import start_redbot

    parser = argparse.ArgumentParser(
        description="Test odometry loop (requires Arduino DONE L R + redbot.get_raw_ticks())."
    )
    parser.add_argument("--wheel-radius", type=float, default=0.065,
                        help="Wheel radius in meters (default: 0.065)")
    parser.add_argument("--ticks-per-rev", type=int, default=192,
                        help="Encoder ticks per revolution (default: 192)")
    parser.add_argument("--wheel-base", type=float, default=0.157,
                        help="Wheel base in meters (default: 0.157)")
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Odometry integration frequency (default: 50 Hz)")
    args = parser.parse_args()

    print("Starting RedBot + odometry test.")
    print("NOTE: Arduino must print 'DONE <L> <R>' and redbot.py must expose get_raw_ticks().")

    start_redbot()
    start_odometry(
        wheel_radius=args.wheel_radius,
        ticks_per_rev=args.ticks_per_rev,
        wheel_base=args.wheel_base,
        loop_hz=args.hz,
    )

    try:
        while True:
            odom = get_odometry_snapshot()
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime(odom.timestamp))}] "
                f"x={odom.x:.3f} m, y={odom.y:.3f} m, "
                f"theta={odom.theta:.3f} rad, v={odom.v:.3f} m/s, "
                f"omega={odom.omega:.3f} rad/s, "
                f"ticksL={odom.ticks_left_raw}, ticksR={odom.ticks_right_raw}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting odometry test.")
