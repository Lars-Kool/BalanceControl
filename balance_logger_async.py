import logging
import logging.config
import time
import re
import copy
import threading

import asyncio

import numpy as np

from scipy.optimize import curve_fit
from simple_pid import PID
from serial_asyncio import open_serial_connection
from Fluigent.SDK import fgt_init, fgt_close, fgt_set_pressure, fgt_get_pressure


logger = logging.getLogger(__name__)


def linear_fit(x, a, b):
    return a * x + b


class Scale:
    """Class for scales."""

    def __init__(self,
                 port: str,
                 baudrate: int = 9600,
                 timeout: int = 1,
                 density: float = 1.0,
                 buffer_length: int = 40,
                 debug: bool = False,
                 PID_params: list[float] = [1.0, 0.0, 0.0],
                 target: float = 0.0,
                 start_value: float = 0.0,
                 output_limits: list[float] = [0, 1000],
                 index: int = 0
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

        self.weight_buffer = np.zeros((self.buffer_length,))
        self.time_buffer = np.zeros((self.buffer_length,))
        self.buffer_id = 0
        self.flowrate = 0.0
        self._data_lock: threading.Lock = threading.Lock()

        self.PID_params = PID_params
        self.target = target / 60
        self.start_value = start_value
        self.output_limits = output_limits
        self.index = index

        logger.debug(f"Generated scale in debug mode: {debug}")
        self.initialized = True

    def start_PID(self) -> None:
        """Start PID controller."""
        logger.info("PID started")
        self.pid = PID(*self.PID_params,
                       setpoint=self.target,
                       starting_output=self.start_value,
                       output_limits=self.output_limits
                       )

    async def run(self) -> None:
        """Actual data collection."""
        self.reader, _ = await open_serial_connection(
            url=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout
        )

        if not self.reader and not self.debug:
            logger.debug(f"Port {self.port} not opened. Stopping run.")
            return

        self.start_time = time.time()
        while True:
            value: float = 0.0
            if self.debug:
                with self._data_lock:
                    dt: float = time.time() - self.time_buffer[self.buffer_id - 1] - self.start_time
                    value = self.weight_buffer[self.buffer_id - 1] * self.density + \
                        dt * self.flowrate * self.density
            else:
                raw_data = await self.reader.readline()
                result = re.search(r"([+-]?\d+\.\d+)", raw_data.decode().strip())
                value = float(result.group(0))

            with self._data_lock:
                self.weight_buffer[self.buffer_id] = value
                self.time_buffer[self.buffer_id] = time.time() - self.start_time
                self.buffer_id = (self.buffer_id + 1) % self.buffer_length

    def get_flowrate(self) -> float:
        """Get flowrate in mL/s.

        Fits weight over time with straight line to obtain flowrate.
        """
        # Make local copy while locking buffers
        with self._data_lock:
            v_asc: np.ndarray = copy.copy(self.weight_buffer)
            t_asc: np.ndarray = copy.copy(self.time_buffer)
            id: int = self.buffer_id

        v_asc = v_asc.take(
            range(id, id + v_asc.size),
            mode="wrap"
        )
        t_asc = t_asc.take(
            range(id, id + t_asc.size),
            mode="wrap"
        )
        dt: float = t_asc[-1] - t_asc[0]
        area: float = np.sum(
            (v_asc[1:] + v_asc[:-1]) * (t_asc[1:] - t_asc[:-1])
        ) / 2 - v_asc[0] * dt
        if (area < 0):
            area *= -1
        return 2 * area / (dt ** 2 * self.density) if dt > 0 else 0

    def get_flowrate_fit(self) -> float:
        """Get flowrate in mL/s using fit."""
        with self._data_lock:
            v: np.ndarray = copy.copy(self.weight_buffer)
            t: np.ndarray = copy.copy(self.time_buffer)

        fit_params, _ = curve_fit(linear_fit, t, v)
        return abs(fit_params[0]) / self.density

    def set_flowrate(self, flowrate: float) -> None:
        """Set flowrate/pressure."""
        if self.debug:
            with self._data_lock:
                self.flowrate = flowrate
        else:
            fgt_set_pressure(self.index, flowrate)

    def get_volume(self) -> float:
        """Return the most recently acquired mass value."""
        with self._data_lock:
            output: float = self.weight_buffer[self.buffer_id - 1]
        return output

    def get_new_pressure(self) -> float:
        """Get new pressure using PID integration."""
        return self.pid(self.get_flowrate_fit())


def log_to_file(volume_water: float,
                flowrate_water: float,
                volume_ethanol: float,
                flowrate_ethanol: float,
                start_time: float,
                pressure_water: float,
                pressure_ethanol: float
                ) -> None:
    """Log values to file."""
    with open("values.txt", 'a') as f:
        f.write(f"{time.time() - start_time:.2f} " +
            f"{volume_water:.2f} {volume_ethanol:.2f} " +
            f"{flowrate_water:.2f} {flowrate_ethanol:.2f} " +
            f"{pressure_water:.2f} {pressure_ethanol:.2f}\n")


def log_to_screen(volume_water: float,
                  flowrate_water: float,
                  volume_ethanol: float,
                  flowrate_ethanol: float
                  ) -> None:
    """Log values to screen."""
    print("\033[H\033[J")
    print("==============================")
    print(f"Weight EtOH    : {volume_ethanol:.2f} g")
    print(f"Weight H2O     : {volume_water:.2f} g")
    print(f"Flowrate EtOH  : {flowrate_ethanol:.2f} mL/min")
    print(f"Flowrate H2O   : {flowrate_water:.2f} mL/min")
    print(f"Total flowrate : {flowrate_water + flowrate_ethanol:.2f} mL/min")
    print(f"Ratio H20/EtOH : {flowrate_water / flowrate_ethanol:.2f}")
    print("==============================")


async def run_control() -> None:
    """Log data on separate thread."""
    start_time = time.time()
    pressure_water: float = 0
    pressure_ethanol: float = 0

    await asyncio.sleep(10)

    scale_water.start_PID()
    scale_ethanol.start_PID()

    while True:
        # Get parameters for logging
        volume_water: float = scale_water.get_volume()
        volume_ethanol: float = scale_ethanol.get_volume()
        flowrate_water: float = scale_water.get_flowrate_fit() * 60  # Convert mL/s to mL/min
        flowrate_ethanol: float = scale_ethanol.get_flowrate_fit() * 60  # Convert mL/s to mL/min

        # Update pressure using pid integration
        pressure_water = scale_water.get_new_pressure()
        pressure_ethanol = scale_ethanol.get_new_pressure()
        scale_water.set_flowrate(pressure_water)
        scale_ethanol.set_flowrate(pressure_ethanol)

        # Log actual pressures
        if not DEBUG:
            pressure_water = fgt_get_pressure(scale_water.index)
            pressure_ethanol = fgt_get_pressure(scale_ethanol.index)

        # Log values to file
        log_to_file(volume_water, flowrate_water,
                    volume_ethanol, flowrate_ethanol,
                    start_time,
                    pressure_water, pressure_ethanol)
        # Log values to screen
        log_to_screen(volume_water, flowrate_water,
                      volume_ethanol, flowrate_ethanol)

        # Update frequency of screen and file
        await asyncio.sleep(0.2)


async def main() -> None:
    """Run experiment."""
    if not scale_water.initialized or not scale_ethanol.initialized:
        logger.error(
            f"Cannot start experiment, as at least one of the scales is not initialized.")
        return

    # Initialize Fluigent library if needed
    if not DEBUG:
        fgt_init()

    # Create tasks to be run asynchronously
    task_water: asyncio.Task = asyncio.create_task(
        scale_water.run()
    )
    task_ethanol: asyncio.Task = asyncio.create_task(
        scale_ethanol.run()
    )
    task_control: asyncio.Task = asyncio.create_task(
        run_control()
    )

    # Run tasks asynchronously
    await task_water
    await task_ethanol
    await task_control

    # Safely close library
    if not DEBUG:
        fgt_close()


if __name__ == "__main__":
    filename: str = "test.txt"
    logging.config.fileConfig("logger_config.toml", defaults={
                              "filename": filename})
    DEBUG: bool = False
    index_water: int = 3
    index_ethanol: int = 2

    water_start_pressure: float = 0.0 if DEBUG else fgt_get_pressure(index_water)
    ethanol_start_pressure: float = 0.0 if DEBUG else fgt_get_pressure(index_ethanol)

    scale_water: Scale = Scale(
        port="COM5", baudrate=9600, timeout=1, density=0.997, debug=DEBUG,
        PID_params=[300, 50, 50], target=150, start_value=water_start_pressure,
        output_limits=[0, 3000], index=index_water
    )
    scale_ethanol: Scale = Scale(
        port="COM6", baudrate=9600, timeout=1, density=0.997, debug=DEBUG,
        PID_params=[2000, 500, 20], target=150, start_value=ethanol_start_pressure,
        output_limits=[0, 5000], index=index_ethanol
    )
    # 0.791

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

