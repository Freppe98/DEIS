#!/usr/bin/env python3
"""
redbot.py — RedBot helper with a reader thread and DONE/ticks parsing.
"""

import os
import re
import time
import threading
import serial

# ================== CONFIG ==================

# Prefer Linux by-id path; allow override via REDBOT_PORT
DEFAULT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10LIMZE-if00-port0"
PORT = os.environ.get("REDBOT_PORT", DEFAULT_PORT)
BAUD = 9600

# Hard-coded test command for main()
L_BUDGET = 200
R_BUDGET = 200
L_SPEED  = 50
R_SPEED  = 50

# ============================================

ser = None
_reader_running = False

_move_done_event = threading.Event()

# Track cumulative ticks for odometry get_raw_ticks()
_ticks_lock = threading.Lock()
_raw_ticks = None  # (L,R) cumulative, signed ints

def get_raw_ticks():
    """Return (leftTicks, rightTicks) cumulative if known, else None."""
    with _ticks_lock:
        return _raw_ticks

def _reader_loop():
    """Background thread: read lines with timeout, optionally log, parse DONE/ticks."""
    global _reader_running, ser, _raw_ticks
    while _reader_running and ser is not None:
        try:
            # Blocking read with timeout avoids OS readiness race seen with in_waiting
            line_bytes = ser.readline()
            if not line_bytes:
                # No data this cycle; brief yield
                time.sleep(0.005)
                continue
            line = line_bytes.decode(errors="replace").strip()
            if not line:
                time.sleep(0.002)
                continue
            print("RX:", line)

            up = line.upper()
            # Accept "DONE" and "DONE ..."
            if up == "DONE" or up.startswith("DONE"):
                _move_done_event.set()
                # parse last two integers as L,R ticks
                try:
                    nums = [int(n) for n in re.findall(r'[-+]?\d+', line)]
                    if len(nums) >= 2:
                        L, R = nums[-2], nums[-1]
                        with _ticks_lock:
                            _raw_ticks = (L, R)
                except Exception:
                    pass
        except Exception as e:
            # when OS signals readable but device returns no bytes
            print(f"[redbot] Reader error: {e}")
            time.sleep(0.05)

def start_redbot():
    """Open serial port and start reader thread (idempotent)."""
    global ser, _reader_running
    if _reader_running:
        return
    if ser is None:
        try:
            print(f"Opening {PORT} @ {BAUD}...", flush=True)
            # exclusive open on POSIX to prevent multiple access to the same port
            try:
                ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2, exclusive=True)
            except TypeError:
                # older pySerial may not support exclusive; fall back
                ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2)
            # Arduino resets on open — give it time
            time.sleep(2.0)
        except Exception as e:
            print(f"Failed to open serial: {e}", flush=True)
            ser = None
            return
    try:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        _reader_running = True
        threading.Thread(target=_reader_loop, daemon=True).start()
    except Exception as e:
        print(f"Error starting reader: {e}", flush=True)

def stop_redbot():
    """Stop reader thread and close serial (idempotent)."""
    global ser, _reader_running
    if not _reader_running and ser is None:
        return
    _reader_running = False
    time.sleep(0.05)
    try:
        if ser and ser.is_open:
            try:
                ser.write(b"stop\n")
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass
    finally:
        ser = None
        print("Serial closed.", flush=True)

def send_ticks(L_budget, R_budget, L_speed, R_speed):
    """Send a single ticksLR command."""
    if ser is None:
        raise RuntimeError("Serial not open. Call start_redbot() first.")
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    _move_done_event.clear()
    cmd = f"ticksLR {int(L_budget)} {int(R_budget)} {int(L_speed)} {int(R_speed)}\n"
    ser.write(cmd.encode("ascii"))
    ser.flush()
    
    print("TX:", cmd.strip())

def send_ticks_blocking(L_budget, R_budget, L_speed, R_speed, timeout=None):
    """Send ticksLR and wait for DONE (True) or timeout (False)."""
    if ser is None:
        raise RuntimeError("Serial not open. Call start_redbot() first.")
    send_ticks(L_budget, R_budget, L_speed, R_speed)
    result = _move_done_event.wait(timeout=timeout)
    if not result:
        try:
            ser.write(b"stop\n")
            ser.flush()

            print("TX: stop (auto on timeout)")
        except Exception:
            pass
    return result

def send_stop():
    """Send stop command."""
    if ser is None:
        raise RuntimeError("Serial not open. Call start_redbot() first.")
    ser.write(b"stop\n")
    ser.flush()
    print("TX: stop")

def is_move_done():
    """Non-blocking check: has RedBot reported DONE?"""
    return _move_done_event.is_set()

def main():
    start_redbot()
    try:
        print("Sending hard-coded ticksLR...")
        send_ticks(L_BUDGET, R_BUDGET, L_SPEED, R_SPEED)
        time.sleep(3.0)
        print("Sending stop...")
        send_stop()
        time.sleep(2.0)
    finally:
        stop_redbot()

if __name__ == "__main__":
    main()
