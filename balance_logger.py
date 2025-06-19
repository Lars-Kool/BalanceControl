import logging
import logging.config
import serial
import time
import re
import asyncio

import numpy as np

from collections import deque
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class Scale:
    port: str = ""
    ser: serial.Serial = None
    initialized: bool = False
    task: asyncio.Task = None
    volume_buffer: deque = deque([0 for i in range(10)], maxlen=10)
    time_buffer: deque = deque([0 for i in range(10)], maxlen=10)

    def __init__(self,
                 port: str,
                 baudrate: int = 9600,
                 timeout: int = 1,
                 density: float = 1.0
                 ):
        """Initialize scale.

        Established serial connection.
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout)
            self.initialized = True
        except Exception as e:
            logger.error(f"Error occurred on port {self.port}: {e}")
        finally:
            self.ser.close()

    def __del__(self):
        """Make sure serial port is properly closed."""
        try:
            self.ser.close()
        except Exception as e:
            logger.error(f"Error occurred on port {self.port}: {e}")

    async def run_task(self):
        """Actual data collection."""
        while True:
            raw_data: str = self.ser.readline().decode().strip()
            value: float = float(re.search(r"([+-]?\d+\.\d+)", raw_data))
            self.volume_buffer.append(value / self.density)
            self.time_buffer.append(time.time())

    def start_task(self):
        """Start data collection task."""
        if not self.initialized:
            logger.error(
                f"Serial connection on port {self.port} not initialized")
            return
        try:
            self.task = asyncio.create_task(self.start_task())
        except Exception as e:
            logger.error(f"Error occurred while running port {self.port}: {e}")
        finally:
            self.task.cancel()
            logger.error("Task automatically cancelled")

    def get_flowrate(self):
        """Get flowrate in g/s.

        Fits weight over time with straight line to obtain flowrate.
        """
        fit_vals, _ = curve_fit(linear_fit,
                                np.array(self.time_buffer),
                                np.array(self.volume_buffer))
        return fit_vals[0]

    def get_volume(self):
        """Return the most recently acquired mass value."""
        return self.data[-1]


def linear_fit(x, a, b) -> np.ndarray:
    return a * x + b


async def main() -> None:
    scale_water: Scale = Scale(
        port="COM3", baudrate=9600, timeout=1, density=0.997)
    scale_ethanol: Scale = Scale(
        port="COM4", baudrate=9600, timeout=1, density=0.791)

    if not scale_water.initialized or not scale_ethanol.initialized:
        logger.error(
            f"Cannot start experiment, as at least one of the scales is not initialized.")
        return

    # Start collecting data
    scale_water.start_task()
    scale_ethanol.start_task()
    asyncio.sleep(5)  # Wait to fill buffer

    logger.info("Time (HH:MM:SS)  Weight water")
    while True:
        flowrate_water: float = scale_water.get_flowrate()
        volume_water: float = scale_water.get_volume()
        flowrate_ethanol: float = scale_ethanol.get_flowrate()
        volume_ethanol: float = scale_ethanol.get_volume()

        logger.info(
            f"{volume_water} {volume_ethanol} {flowrate_water} {flowrate_ethanol} {time.time()}"
        )


if __name__ == "__main__":
    filename: str = "test.txt"
    logging.config.fileConfig("logger_config.toml", defaults={
                              "filename": filename})
    asyncio.run(main())
