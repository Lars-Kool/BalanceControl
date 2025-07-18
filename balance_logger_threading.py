import logging
import logging.config
import serial
import time
import re
import threading
import copy

import numpy as np

from simple_pid import PID
from Fluigent.SDK import fgt_init, fgt_close, fgt_set_pressure, fgt_get_pressure


logger = logging.getLogger(__name__)


class Scale:
    """Class for scales."""

    port: str = ""
    baudrate: int
    timeout: int
    density: float
    debug: bool

    ser: serial.Serial = None
    start_time: float
    initialized: bool = False

    buffer_id: int
    weight_buffer: np.ndarray
    time_buffer: np.ndarray
    _data_lock: threading.Lock = threading.Lock()
    flowrate: float
    pid: PID
    target: float

    def __init__(self,
                 port: str,
                 baudrate: int = 9600,
                 timeout: int = 1,
                 density: float = 1.0,
                 buffer_length: int = 10,
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
        self.flowrate = 0

        self.PID_params = PID_params
        self.target = target / 60
        self.start_value = start_value
        self.output_limits = output_limits
        self.index = index

        logger.debug(f"Generated scale in debug mode: {debug}")
        try:
            if not self.debug:
                self.ser = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=timeout
                )
            self.initialized = True
        except Exception as e:
            logger.error(f"Error occurred on port {self.port}: {e}")

    def __del__(self) -> None:
        """Make sure serial port is properly closed."""
        try:
            if not self.debug:
                self.ser.close()
        except Exception as e:
            logger.error(f"Error occurred while closing port {self.port}: {e}")

    def start_PID(self) -> None:
        """Start PID controller."""
        logger.info("PID started")
        self.pid = PID(*self.PID_params, setpoint=self.target, starting_output=self.start_value,
                       output_limits=self.output_limits
                       )

    def run_on_thread(self) -> None:
        """Actual data collection."""
        if not self.ser and not self.debug:
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
                raw_data = self.ser.readline().decode().strip()
                print(self.ser.in_waiting)
                result = re.search(r"([+-]?\d+\.\d+)", raw_data)
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
        return self.pid(self.get_flowrate())


def log_to_file(
                volume_water: float,
                flowrate_water: float,
                volume_ethanol: float,
                flowrate_ethanol: float,
                start_time: float
                ) -> None:
    """Log values to file."""
    with open("values.txt", 'a') as f:
        f.write(f"{time.time() - start_time:.2f} {volume_water:.2f} {volume_ethanol:.2f} {flowrate_water:.2f} {flowrate_ethanol:.2f}\n")


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


def main_thread() -> None:
    """Log data on separate thread."""
    start_time = time.time()
    pressure_water: float = 0
    pressure_ethanol: float = 0

    time.sleep(5)

    scale_water.start_PID()
    scale_ethanol.start_PID()

    while True:
        # Get parameters for logging
        volume_water: float = scale_water.get_volume()
        volume_ethanol: float = scale_ethanol.get_volume()
        flowrate_water: float = scale_water.get_flowrate() * 60  # Convert mL/s to mL/min
        flowrate_ethanol: float = scale_ethanol.get_flowrate() * 60  # Convert mL/s to mL/min

        # Log values to file
        log_to_file(volume_water, flowrate_water, volume_ethanol, flowrate_ethanol, start_time)
        # Log values to screen
        # log_to_screen(volume_water, flowrate_water, volume_ethanol, flowrate_ethanol)

        # Update pressure using pid integration
        pressure_water = scale_water.get_new_pressure()
        pressure_ethanol = scale_ethanol.get_new_pressure()
        scale_water.set_flowrate(pressure_water)
        scale_ethanol.set_flowrate(pressure_ethanol)

        time.sleep(1)

def main() -> None:
    """Run experiment."""
    if not scale_water.initialized or not scale_ethanol.initialized:
        logger.error(
            f"Cannot start experiment, as at least one of the scales is not initialized.")
        return

    if not DEBUG:
        fgt_init()

    water_thread = threading.Thread(
        target=scale_water.run_on_thread,
        daemon=True
    )
    water_thread.start()

    ethanol_thread = threading.Thread(
        target=scale_ethanol.run_on_thread,
        daemon=True
    )
    ethanol_thread.start()

    main_loop = threading.Thread(
        target=main_thread,
        daemon=True
    )
    main_loop.start()

    while True:
        pass

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
        PID_params=[100, 50, 0], target=50, start_value=water_start_pressure,
        output_limits=[0, 2000], index=index_water
    )
    scale_ethanol: Scale = Scale(
        port="COM6", baudrate=9600, timeout=1, density=0.791, debug=DEBUG,
        PID_params=[100, 50, 0], target=50, start_value=ethanol_start_pressure,
        output_limits=[0, 2000], index=index_ethanol
    )

    main()
