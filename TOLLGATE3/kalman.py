"""kalman.py

Lightweight Kalman fusion for (x,y,theta) where odometry provides prediction
and world_perception supplies x,y measurements when visible.
Units: meters (theta radians).
"""

from __future__ import annotations

from typing import Tuple
import math

_X = [0.0, 0.0, 0.0]         # state: x, y, theta
_P = [ [0.05,0,0], [0,0.05,0], [0,0,0.02] ]  # covariance
_Q = [ [0.001,0,0], [0,0.001,0], [0,0,0.0005] ]  # process noise
_R = [ [0.01,0], [0,0.01] ]  # measurement noise for x,y
_DEBUG = bool(int(__import__('os').environ.get('KALMAN_DEBUG', '0')))


def init_kalman(initial_x: float, initial_y: float, initial_theta: float):
    global _X, _P
    _X = [initial_x, initial_y, initial_theta]
    _P = [ [0.01,0,0], [0,0.01,0], [0,0,0.01] ]


def kalman_predict_from_odom(odom_state, dt: float):
    """Use odometry snapshot as direct prediction (assumes it already integrated)."""
    global _X, _P
    if odom_state is None:
        return
    # Direct overwrite prediction (could integrate velocities but odom did already)
    _X[0] = getattr(odom_state, 'x', _X[0])
    _X[1] = getattr(odom_state, 'y', _X[1])
    _X[2] = getattr(odom_state, 'theta', _X[2])
    # Increase covariance with process noise Q
    for i in range(3):
        for j in range(3):
            _P[i][j] += _Q[i][j]
    if _DEBUG:
        try:
            print(f"[KF] predict odom x={getattr(odom_state,'x',None):.3f} y={getattr(odom_state,'y',None):.3f} th={getattr(odom_state,'theta',None):.3f}")
        except Exception:
            pass


def kalman_update_from_world(world_state):
    """Measurement update using world_state.robot_world_mm if visible."""
    global _X, _P
    if world_state is None or not getattr(world_state, 'robot_visible', False):
        return
    mm = world_state.robot_world_mm
    if not mm:
        return
    zx = mm[0] / 1000.0  # mm->m
    zy = mm[1] / 1000.0
    # Measurement matrix H: only x,y
    H = [ [1,0,0], [0,1,0] ]
    # y = z - Hx
    y_vec = [ zx - (_X[0]), zy - (_X[1]) ]
    # S = H P H^T + R (2x2)
    S00 = H[0][0]*_P[0][0] + H[0][1]*_P[1][0] + H[0][2]*_P[2][0] + _R[0][0]
    S01 = H[0][0]*_P[0][1] + H[0][1]*_P[1][1] + H[0][2]*_P[2][1] + _R[0][1]
    S10 = H[1][0]*_P[0][0] + H[1][1]*_P[1][0] + H[1][2]*_P[2][0] + _R[1][0]
    S11 = H[1][0]*_P[0][1] + H[1][1]*_P[1][1] + H[1][2]*_P[2][1] + _R[1][1]
    # Invert S (2x2)
    det = S00*S11 - S01*S10
    if abs(det) < 1e-9:
        return
    invS00 =  S11/det
    invS01 = -S01/det
    invS10 = -S10/det
    invS11 =  S00/det
    # K = P H^T S^-1 (3x2)
    # Compute PH^T first (3x2)
    PHt = [
        [ _P[0][0], _P[0][1] ],
        [ _P[1][0], _P[1][1] ],
        [ _P[2][0], _P[2][1] ],
    ]
    K = [
        [ PHt[0][0]*invS00 + PHt[0][1]*invS10, PHt[0][0]*invS01 + PHt[0][1]*invS11 ],
        [ PHt[1][0]*invS00 + PHt[1][1]*invS10, PHt[1][0]*invS01 + PHt[1][1]*invS11 ],
        [ PHt[2][0]*invS00 + PHt[2][1]*invS10, PHt[2][0]*invS01 + PHt[2][1]*invS11 ],
    ]
    # Update state X = X + K y
    _X[0] += K[0][0]*y_vec[0] + K[0][1]*y_vec[1]
    _X[1] += K[1][0]*y_vec[0] + K[1][1]*y_vec[1]
    _X[2] += K[2][0]*y_vec[0] + K[2][1]*y_vec[1]
    if _DEBUG:
        try:
            print(f"[KF] update meas x={zx:.3f} y={zy:.3f} | fused x={_X[0]:.3f} y={_X[1]:.3f} th={_X[2]:.3f}")
        except Exception:
            pass
    # Update covariance P = (I - K H) P
    KH = [
        [K[0][0]*H[0][0] + K[0][1]*H[1][0], K[0][0]*H[0][1] + K[0][1]*H[1][1], K[0][0]*H[0][2] + K[0][1]*H[1][2]],
        [K[1][0]*H[0][0] + K[1][1]*H[1][0], K[1][0]*H[0][1] + K[1][1]*H[1][1], K[1][0]*H[0][2] + K[1][1]*H[1][2]],
        [K[2][0]*H[0][0] + K[2][1]*H[1][0], K[2][0]*H[0][1] + K[2][1]*H[1][1], K[2][0]*H[0][2] + K[2][1]*H[1][2]],
    ]
    for i in range(3):
        for j in range(3):
            val = _P[i][j]
            for k in range(3):
                val -= KH[i][k]*_P[k][j]
            _P[i][j] = val


def get_kalman_state() -> Tuple[float,float,float]:
    return _X[0], _X[1], _X[2]


if __name__ == "__main__":
    class Odom: pass
    o = Odom(); o.x=0.1; o.y=0.05; o.theta=0.0
    init_kalman(0,0,0)
    kalman_predict_from_odom(o, 0.02)
    class WS: pass
    ws = WS(); ws.robot_visible=True; ws.robot_world_mm=(120.0, 80.0)
    kalman_update_from_world(ws)
    print("State:", get_kalman_state())
