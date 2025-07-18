import logging
import logging.config
import time
import serial
import re

import numpy as np

from Fluigent.SDK import fgt_init, fgt_close, fgt_get_controllersInfo, fgt_set_pressure, fgt_get_pressure


def test_serial(port: str) -> None:
    fgt_init()

    fgt_set_pressure(2, 1000)
    with serial.Serial(port=port, baudrate=9600, timeout=1) as ser:
        i: int = 0
        values: list[float] = [0 for i in range(10)]
        times: list[float] = [0 for i in range(10)]
        while True:
            raw_data: str = ser.readline().decode().strip()
            result = re.search(r"([+-]?\d+\.\d+)", raw_data)
            values[i] = float(result.group(0))
            times[i] = time.perf_counter()
            if i == 0:
                print(f"dt: {times[0] - times[1]}, weight: {values[0]}")
            i = (i + 1) % 10

def test_fluigent() -> None:
    fgt_init()

    channels = fgt_get_controllersInfo()
    print(channels)

    fgt_set_pressure(0, 0)

    time.sleep(5)

    print(f"Pressure: {fgt_get_pressure(0):.2f} mbar")

    fgt_set_pressure(0, 0)

    time.sleep(1)

    fgt_close()


if __name__ == "__main__":
    test_serial("COM6")
    # test_fluigent()
