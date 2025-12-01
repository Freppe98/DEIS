#!/usr/bin/env python3
"""
collision_sensor.py

Simple collision monitor for GPIO24.

- LOW  = collision detected
- HIGH   = no collision

Can be:
  1) Imported and run as a background thread in your crab_main.py
  2) Executed directly to test the sensor from the terminal
"""

import threading
import time
import os

# Lean version: RPi.GPIO only (requires /dev/gpiomem permissions)

try:
    import RPi.GPIO as GPIO  # type: ignore
    _GPIO_IMPORT_ERROR = None
except Exception as _e:
    GPIO = None  # type: ignore
    _GPIO_IMPORT_ERROR = _e


# --- CONFIG ---

COLLISION_PIN = int(os.getenv("T3_COLLISION_PIN", "24"))  # BCM numbering
POLL_INTERVAL = float(os.getenv("T3_COLLISION_POLL_SEC", "0.1"))  # seconds
ENABLE_SENSOR = os.getenv("T3_COLLISION_ENABLE", "1") == "1"


# --- INTERNAL STATE ---

_collision_event = threading.Event()
_monitor_thread = None
_monitor_stop = threading.Event()
_gpio_initialized = False
_pigpio_initialized = False
_gpiod_initialized = False
_state_lock = threading.Lock()
_disabled_reason = None


def _setup_gpio():
    global _gpio_initialized
    with _state_lock:
        global _disabled_reason

        if not ENABLE_SENSOR:
            _disabled_reason = "disabled via T3_COLLISION_ENABLE=0"
            return

        if _gpio_initialized:
            return

        # Lean path: use RPi.GPIO directly

        if GPIO is None:
            if not _gpio_initialized:
                if _disabled_reason is None:
                    _disabled_reason = f"GPIO unavailable ({_GPIO_IMPORT_ERROR})"
            return

        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            # Assuming sensor actively drives LOW when blocked, HIGH otherwise.
            # Use pull-up so default = HIGH if floating.
            GPIO.setup(COLLISION_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            _gpio_initialized = True
        except RuntimeError as e:
            # e.g., "No access to /dev/mem. Try running as root!"
            _disabled_reason = str(e)
            _gpio_initialized = False
        except Exception as e:
            _disabled_reason = f"GPIO setup failed: {e}"
            _gpio_initialized = False


def _monitor_loop():
    """Internal thread: polls GPIO and updates _collision_event."""
    last_state = None
    while not _monitor_stop.is_set():
        level = GPIO.input(COLLISION_PIN)
        is_low = (level == GPIO.LOW)

        if is_low:
            _collision_event.set()
            new_state = True
        else:
            _collision_event.clear()
            new_state = False

        last_state = new_state
        time.sleep(POLL_INTERVAL)


# --- PUBLIC API (for crab_main.py) ---

def start_collision_monitor():
    """
    Initialize GPIO and start the collision monitoring thread.

    Call this once from crab_main.py, e.g.:

        from collision_sensor import start_collision_monitor, is_collision

        start_collision_monitor()

        while True:
            if is_collision():
                print("Collision!")
            time.sleep(0.05)
    """
    global _monitor_thread
    _setup_gpio()

    if not _gpio_initialized:
        # Graceful no-op when neither backend available
        reason = _disabled_reason or "unknown"
        print(f"[collision] Disabled (mock mode): {reason}")
        return

    print("[collision] Using RPi.GPIO backend.")

    if _monitor_thread is None or not _monitor_thread.is_alive():
        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            name="CollisionMonitor",
            daemon=True,
        )
        _monitor_thread.start()


def is_collision() -> bool:
    """Returns True if COLLISION_PIN signals collision. False when uninitialized."""
    if not _gpio_initialized:
        return False
    return _collision_event.is_set()


def stop_collision_monitor():
    """Signal the monitor thread to stop and join briefly."""
    _monitor_stop.set()
    try:
        if _monitor_thread and _monitor_thread.is_alive():
            _monitor_thread.join(timeout=1.0)
    except Exception:
        pass


def cleanup():
    """
    Clean up GPIO. Call this at program exit if you want.
    (Not strictly required if you only run as daemon thread.)
    """
    with _state_lock:
        try:
            stop_collision_monitor()
        except Exception:
            pass
        if _gpio_initialized and GPIO is not None:
            try:
                GPIO.cleanup()
            except Exception:
                pass


# --- TEST MAIN (run this file directly to test the sensor) ---

if __name__ == "__main__":
    print("Starting collision_sensor test on GPIO24 (BCM mode).")
    print("LOW = collision, HIGH = no collision.")
    print("Press Ctrl+C to exit.")

    try:
        start_collision_monitor()
        last = None
        while True:
            c = is_collision()
            if c != last:
                print(f"Collision: {c}")
                last = c
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cleanup()
