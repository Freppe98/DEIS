#!/usr/bin/env python3
import threading
import time

try:
    from smbus2 import SMBus
except ImportError:
    # if smbus2 isn't installed
    from smbus import SMBus

INA260_I2C_ADDR = 0x40

REG_CONFIG      = 0x00
REG_CURRENT     = 0x01
REG_BUS_VOLTAGE = 0x02
REG_POWER       = 0x03

# LSB sizes datasheet
CURRENT_LSB_A = 0.00125   # 1.25 mA per bit
VOLTAGE_LSB_V = 0.00125   # 1.25 mV per bit

class INA260Reader(threading.Thread):
    def __init__(self, bus_num=1, address=INA260_I2C_ADDR, interval=0.2):
        super().__init__(daemon=True)
        self.bus = SMBus(bus_num)
        self.address = address
        self.interval = interval

        self._lock = threading.Lock()
        self._current_A = 0.0
        self._voltage_V = 0.0
        self._running = True

    def _read_register_16(self, reg):
        """Read 16-bit register (big-endian) and return as int."""
        raw = self.bus.read_word_data(self.address, reg)
        # SMBus is little-endian; INA260 is big-endian → swap bytes
        value = ((raw & 0xFF) << 8) | (raw >> 8)
        return value

    def _to_signed_16(self, value):
        """Convert 16-bit unsigned to signed."""
        if value & 0x8000:
            value -= 1 << 16
        return value

    def run(self):
        while self._running:
            try:
                # Read current
                raw_current = self._read_register_16(REG_CURRENT)
                signed_current = self._to_signed_16(raw_current)
                current_A = signed_current * CURRENT_LSB_A

                # Read bus voltage
                raw_voltage = self._read_register_16(REG_BUS_VOLTAGE)
                voltage_V = raw_voltage * VOLTAGE_LSB_V

                with self._lock:
                    self._current_A = current_A
                    self._voltage_V = voltage_V

            except Exception as e:
                print(f"[INA260] Read error: {e}")

            time.sleep(self.interval)

    def stop(self):
        self._running = False
        try:
            self.bus.close()
        except Exception:
            pass

    def get_current(self):
        with self._lock:
            return self._current_A

    def get_voltage(self):
        with self._lock:
            return self._voltage_V


# ---- global wrapper API  ----

_ina260_thread = None

def startINA260(bus_num=1, address=INA260_I2C_ADDR, interval=0.2):
    global _ina260_thread
    if _ina260_thread is None:
        _ina260_thread = INA260Reader(bus_num=bus_num, address=address, interval=interval)
        _ina260_thread.start()

def stopINA260():
    global _ina260_thread
    if _ina260_thread is not None:
        _ina260_thread.stop()
        _ina260_thread = None

def getCurrent():
    if _ina260_thread is None:
        raise RuntimeError("INA260 thread not started. Call startINA260() first.")
    return _ina260_thread.get_current()

def getVoltage():
    if _ina260_thread is None:
        raise RuntimeError("INA260 thread not started. Call startINA260() first.")
    return _ina260_thread.get_voltage()


# ---- Test main ----

if __name__ == "__main__":
    print("Starting INA260 reader…")
    startINA260(bus_num=1, address=INA260_I2C_ADDR, interval=0.2)

    try:
        while True:
            I = getCurrent()
            V = getVoltage()
            P = V * I
            print(f"Voltage: {V:6.3f} V  |  Current: {I:7.3f} A  |  Power (approx): {P:7.3f} W")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping INA260 reader…")
    finally:
        stopINA260()
        print("Done.")
