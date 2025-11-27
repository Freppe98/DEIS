#!/usr/bin/env python3
"""
redbot_thread_simple.py — minimal RedBot helper with a reader thread.

- Change PORT/BAUD and the hard-coded command in main().
- Run with:  python3 redbot_thread_simple.py
"""

import time
import threading
import serial

# ================== CONFIG ==================

PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10LIMZE-if00-port0"   # e.g. "/dev/ttyACM0" or "/dev/serial/by-id/usb-..."
BAUD = 9600

# Hard-coded test command for main()
L_BUDGET = 200
R_BUDGET = 200
L_SPEED  = 150
R_SPEED  = 150

# ============================================

ser = None
_reader_running = False

_move_done_event = threading.Event()

def _reader_loop():
    """Background thread: just print anything the RedBot sends."""
    global _reader_running, ser
    while _reader_running and ser is not None:
        if ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                print("RX:", line)
                # Accept either exactly "DONE" or lines that start with "DONE"
                # (e.g. "DONE ticksLR") so the move-done event reliably fires.
                l = line.strip().upper()
                if l == "DONE" or l.startswith("DONE ") or l.startswith("DONE\t") or l == "DONE":
                    _move_done_event.set()
        else:
            time.sleep(0.01)


def start_redbot():
    """Open serial port and start reader thread (idempotent)."""
    global ser, _reader_running
    import serial, threading, time
    if _reader_running:
        return
    # if serial already open, just start reader
    if ser is None:
        try:
            print(f"Opening {PORT} @ {BAUD}...", flush=True)
            ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2)
            # give Arduino time to reset
            time.sleep(2.0)
        except Exception as e:
            print(f"Failed to open serial: {e}", flush=True)
            ser = None
            return
    try:
        # clear any leftover data
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
    import time
    if not _reader_running and ser is None:
        return
    _reader_running = False
    # give reader thread time to exit and flush
    time.sleep(0.05)
    try:
        # best-effort stop command to robot
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

    # Remove any previous pending input so stale "DONE" lines don't
    # immediately satisfy the next wait.
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    _move_done_event.clear() ######### EVENT

    cmd = f"ticksLR {int(L_budget)} {int(R_budget)} {int(L_speed)} {int(R_speed)}\n"
    ser.write(cmd.encode("ascii"))
    ser.flush()
    print("TX:", cmd.strip())


def send_ticks_blocking(L_budget, R_budget, L_speed, R_speed, timeout=None):
    """Send `ticksLR` then block until the RedBot reports DONE or timeout.

    - `timeout` in seconds or None to wait indefinitely.
    - Returns True if DONE received, False on timeout.
    """
    send_ticks(L_budget, R_budget, L_speed, R_speed)
    result = _move_done_event.wait(timeout=timeout)
    if not result:
        # Timed out waiting for DONE — stop motors to avoid runaway.
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
        # === Manually hard-code commands here ===
        print("Sending hard-coded ticksLR...")
        send_ticks(L_BUDGET, R_BUDGET, L_SPEED, R_SPEED)

        # Let it run a bit
        time.sleep(3.0)

        print("Sending stop...")
        send_stop()

        # Keep reading any last responses
        time.sleep(2.0)

    finally:
        stop_redbot()


if __name__ == "__main__":
    main()
