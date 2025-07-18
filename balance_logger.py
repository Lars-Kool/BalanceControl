import logging
import logging.config
import serial
import time
import re
import asyncio

import numpy as np

from asyncio import get_event_loop
from serial_asyncio import open_serial_connection

logger = logging.getLogger(__name__)


def linear_fit(x, a, b) -> np.ndarray:
    """Fit for y = ax + b."""
    return a * x + b


class Scale:
    """Class for scales."""

    port: str = ""
    baudrate: int = 0
    timeout: int = 0
    density: float = 1.0
    debug: bool = False

    ser: serial.Serial = None
    initialized: bool = False

    task: asyncio.Task = None
    buffer_id: int = 0
    volume_buffer: np.ndarray = np.zeros((10,))
    time_buffer: np.ndarray = np.zeros((10,))
    start_time: float = 0.0

    def __init__(self,
                 port: str,
                 baudrate: int = 9600,
                 timeout: int = 1,
                 density: float = 1.0,
                 buffer_length: int = 10,
                 debug: bool = False
                 ) -> None:
        """Initialize scale.

        Established serial connection.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.density = density
        self.buffer_length = buffer_length
        self.debug = debug

        logger.debug(f"Generated scale in debug mode: {debug}")
        try:
            if not self.debug:
                self.ser = serial.Serial(port, baudrate, timeout)
            self.initialized = True
        except Exception as e:
            logger.error(f"Error occurred on port {self.port}: {e}")

    def __del__(self) -> None:
        """Make sure serial port is properly closed."""
        try:
            if not self.debug:
                self.ser.close()
        except Exception as e:
            logger.error(f"Error occurred on port {self.port}: {e}")

        if self.task:
            self.task.cancel()
            logger.info("Task cancelled as part of destructor.")

    async def run_task(self) -> None:
        """Actual data collection."""
        self.start_time = time.perf_counter()
        while True:
            value: float = 0.0
            if self.debug:
                value = time.perf_counter() - self.start_time
            else:
                raw_data: str = self.ser.readline().decode().strip()
                value = float(
                    re.search(r"([+-]?\d+\.\d+)", raw_data)
                )

            self.volume_buffer[self.buffer_id] = value / self.density
            self.time_buffer[self.buffer_id] = time.perf_counter()
            logger.debug(self.volume_buffer[self.buffer_id])
            self.buffer_id = (self.buffer_id + 1) % self.buffer_length
            await asyncio.sleep(0.1)

    def start_task(self) -> None:
        """Start data collection task."""
        if not self.initialized:
            logger.error(
                f"Serial connection on port {self.port} not initialized")
            return
        try:
            logger.debug("Creating task.")
            self.task = asyncio.create_task(self.run_task())
        except Exception as e:
            logger.error(f"Error occurred while running port {self.port}: {e}")

    async def get_flowrate(self) -> float:
        """Get flowrate in g/s.

        Fits weight over time with straight line to obtain flowrate.
        """
        v_asc: np.ndarray = self.volume_buffer.take(
            range(self.buffer_id, self.buffer_id + self.buffer_length - 1),
            mode="wrap"
        )
        t_asc: np.ndarray = self.time_buffer.take(
            range(self.buffer_id, self.buffer_id + self.buffer_length - 1),
            mode="wrap"
        )
        dt: float = t_asc[-1] - t_asc[0]
        area: float = np.sum(
            (v_asc[1:] + v_asc[:-1]) * (t_asc[1:] - t_asc[:-1])
        ) / 2 - v_asc[0] * dt
        logger.debug(f"Area: {area}")
        return 2 * area / (dt ** 2) if dt > 0 else 0

    async def get_volume(self) -> float:
        """Return the most recently acquired mass value."""
        return self.volume_buffer[self.buffer_id - 1]


async def main_loop():
    while True:
        volume_water: float = await scale_water.get_volume()
        volume_ethanol: float = await scale_ethanol.get_volume()
        flowrate_water: float = await scale_water.get_flowrate()
        flowrate_ethanol: float = await scale_ethanol.get_flowrate()

        logger.debug(
            f"{volume_water:.2f} " +
            f"{volume_ethanol:.2f} " +
            f"{flowrate_water:.2f} " +
            f"{flowrate_ethanol:.2f} " +
            f"{time.time():.2f}"
        )
        await asyncio.sleep(1)


async def main() -> None:
    """Run main function asynchronously."""
    # Start collecting data
    scale_ethanol.start_task()
    scale_water.start_task()
    task: asyncio.Task = asyncio.create_task(main_loop())
    await asyncio.gather(scale_water.task, scale_ethanol.task, task)


if __name__ == "__main__":
    filename: str = "test.txt"
    logging.config.fileConfig("logger_config.toml", defaults={
                              "filename": filename})

    scale_water: Scale = Scale(
        port="COM3", baudrate=9600, timeout=1, density=10.997, debug=True)
    scale_ethanol: Scale = Scale(
        port="COM4", baudrate=9600, timeout=1, density=0.791, debug=True)

    if not scale_water.initialized or not scale_ethanol.initialized:
        logger.error(
            f"Cannot start experiment, as at least one of the scales is not initialized.")
    else:
        asyncio.run(main())
