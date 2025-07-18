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
    volume_buffer: np.ndarray
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
                 start_value: float = 0.0
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

        self.volume_buffer = np.zeros((self.buffer_length,))
        self.time_buffer = np.zeros((self.buffer_length,))
        self.buffer_id = 0
        self.flowrate = 0

        logger.debug(f"Setpoint = {target:.2f}")
        self.pid = PID(*PID_params, setpoint=target, starting_output=start_value)
        self.pid.auto_mode = False

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

        fgt_set_pressure(index, 0)


    def run_on_thread(self) -> None:
        """Actual data collection."""
        if not self.ser and not self.debug:
            logger.debug(f"Port {self.port} not opened. Stopping run.")
            return

        self.start_time = time.time()
        init_val: float = self.get_mass()
        self.volume_buffer = np.ones((self.buffer_length, )) * init_val
        self.time_buffer = np.ones((self.buffer_length, )) * self.start_time
        while True:
            value: float = 0.0
            if self.debug:
                with self._data_lock:
                    dt: float = time.time() - self.time_buffer[self.buffer_id - 1] - self.start_time
                    value = self.volume_buffer[self.buffer_id - 1] * self.density + dt * self.flowrate
            else:
                value = self.get_mass()

            with self._data_lock:
                self.volume_buffer[self.buffer_id] = value / self.density
                self.time_buffer[self.buffer_id] = time.time() - self.start_time
                self.buffer_id = (self.buffer_id + 1) % self.buffer_length
            time.sleep(0.1)

    def get_mass(self) -> float:
        raw_data: str = self.ser.readline().decode().strip()
        result = re.search(r"([+-]?\d+\.\d+)", raw_data)
        value = float(result.group(0))
        return value


    def get_flowrate(self) -> float:
        """Get flowrate in g/s.

        Fits weight over time with straight line to obtain flowrate.
        """
        # Make local copy while locking buffers
        with self._data_lock:
            v_asc: np.ndarray = copy.copy(self.volume_buffer)
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
        logger.debug(v_asc)
        logger.debug(t_asc)
        return -2 * area / (dt ** 2) if dt > 0 else 0

    def set_flowrate(self, flowrate: float) -> None:
        """Set flowrate, debug use only."""
        if self.debug:
            with self._data_lock:
                self.flowrate = flowrate

    def get_volume(self) -> float:
        """Return the most recently acquired mass value."""
        with self._data_lock:
            output: float = self.volume_buffer[self.buffer_id - 1]
        return output

    def get_new_pressure(self) -> float:
        """Get new pressure using PID integration."""
        logger.debug(self.pid.components)
        return self.pid(self.get_flowrate() * 60)  # Convert mL/s to mL/min

    def start_pid(self) -> None:
        self.pid.auto_mode = True

def log_to_file(volume: float,
                flowrate: float,
                pressure: float,
                start_time: float
                ) -> None:
    """Log values to file."""
    with open("values.txt", 'a') as f:
        f.write(f"{time.time() - start_time:.2f} {volume:.2f} {flowrate:.2f} {pressure:.2f}\n")


def log_to_screen(volume: float,
                  flowrate: float,
                  pressure: float
                  ) -> None:
    """Log values to screen."""
    print("\033[H\033[J")
    print("==============================")
    print(f"Volume    : {volume:.2f} ml")
    print(f"Flowrate  : {flowrate:.2f} mL/min")
    print(f"P_imposed : {pressure:.2f} mbar")
    print(f"P_measured: {fgt_get_pressure(index):.2f}")
    print("==============================")


def main_thread() -> None:
    """Log data on separate thread."""
    start_time = time.time()
    pressure: float = fgt_get_pressure(index)

    scale.start_pid()

    try:
        while True:
            # Get parameters for logging
            volume: float = scale.get_volume()
            flowrate: float = scale.get_flowrate() * 60 # Convert mL/s to mL/min

            # Log values to file
            log_to_file(volume, flowrate, pressure, start_time)
            # Log values to screen
            log_to_screen(volume, flowrate, pressure)

            # Update pressure using pid integration
            pressure = scale.get_new_pressure()
            if DEBUG:
                scale.set_flowrate(pressure)
            else:
                fgt_set_pressure(index, pressure)

            time.sleep(0.1)
    except KeyboardInterrupt:
        fgt_set_pressure(index, 0)


def main() -> None:
    """Run experiment."""
    if not scale.initialized:
        logger.error(
            f"Cannot start experiment, as at least one of the scales is not initialized.")
        return

    if not DEBUG:
        fgt_init()

    thread_pid = threading.Thread(
        target=scale.run_on_thread,
        daemon=True
    )
    thread_pid.start()

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
    index: int = 0

    fgt_set_pressure(index, 30)

    start_pressure: float = 1.0 if DEBUG else fgt_get_pressure(index)

    scale: Scale = Scale(
        port="COM7", baudrate=9600, timeout=1, density=0.997, debug=DEBUG,
        PID_params=[0.3, 1.0, 0], target=100, start_value=start_pressure)

    main()
