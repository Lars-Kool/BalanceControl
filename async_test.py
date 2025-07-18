import logging
import logging.config
import time
import serial
import re
import asyncio

import numpy as np

from serial_asyncio import open_serial_connection


async def test_serial(port: str) -> None:
    reader, _ = await open_serial_connection(
        url = port,
        baudrate=9600,
        timeout=1
    )
    i: int = 0
    values: list[float] = [0 for i in range(10)]
    times: list[float] = [0 for i in range(10)]
    while True:
        raw_data: str = await reader.readline()
        result = re.search(r"([+-]?\d+\.\d+)", raw_data.decode().strip())
        values[i] = float(result.group(0))
        times[i] = time.perf_counter()
        if i == 0:
            print(f"dt: {times[0] - times[1]}, weight: {values[0]}")
        i = (i + 1) % 10

async def main() -> None:
    task1: asyncio.Task = asyncio.create_task(
        test_serial("COM5")
    )
    task2: asyncio.Task = asyncio.create_task(
        test_serial("COM6")
    )

    await task1
    await task2

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
