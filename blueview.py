# DEPENDENCIES
# pip3 install bleak construct
# pip install bleak filterpy rich

import asyncio
import time
import numpy as np
import subprocess

from bleak import BleakScanner
from filterpy.kalman import KalmanFilter

from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

PATH_LOSS_EXPONENT = 2.8
RANGE = 50.0
WARN = 5.0
DEVICE_TIMEOUT = 30.0

PREFERRED_ADAPTER_MAC = "F0:09:0D:E9:9B:FC"


def get_adapter_hci_by_mac(mac_address):
    try:
        output = subprocess.check_output(["hciconfig"], text=True)
    except subprocess.CalledProcessError:
        return None

    adapters = output.strip().split("\n\n")
    for block in adapters:
        lines = block.splitlines()
        if not lines:
            continue
        hci_line = lines[0]
        hci_name = hci_line.split(":")[0]
        for line in lines:
            if "BD Address" in line and mac_address in line:
                return hci_name
    return None


def estimate_distance(rssi, tx_power=-59, n=PATH_LOSS_EXPONENT):
    if rssi == 0 or tx_power is None:
        return None
    ratio = (tx_power - rssi) / (10 * n)
    return round(10**ratio, 2)


APPLE_ID = 0x004C
GOOGLE_ID = 0x00E0


def decode_manufacturer_data(manufacturer_data: dict[int, bytes]) -> str:
    output = []

    for company_id, data in manufacturer_data.items():
        company = {
            0x004C: "Apple",
            0x00E0: "Google",
            0x0006: "Microsoft",
            0x0131: "Samsung",
            0x0075: "Huawei",
            0x00D2: "Fitbit",
        }.get(company_id, f"Unknown (0x{company_id:04x})")

        # line = f"{company}: {data.hex()}"
        line = f"{company}"

        # Apple iBeacon decoding
        if company_id == APPLE_ID and data.startswith(b"\x02\x15") and len(data) >= 23:
            uuid = data[2:18]
            major = int.from_bytes(data[18:20], "big")
            minor = int.from_bytes(data[20:22], "big")
            tx_power = int.from_bytes(data[22:23], "big", signed=True)

            line += f" [iBeacon] UUID: {uuid.hex()}, Major: {major}, Minor: {minor}, Tx: {tx_power}dBm"

        output.append(line)

    if output:
        return "\n".join(output)
    else:
        return "—"


def decode_service_data(service_data: dict[str, bytes]) -> str:
    output = []

    for uuid, data in service_data.items():
        label = f"{uuid}"
        line = ""

        # Example: Try decoding heart rate from standard UUID
        if uuid.lower() == "0000180d-0000-1000-8000-00805f9b34fb" and len(data) > 1:
            hr_value = data[1]
            line = f"{label}: Heart Rate = {hr_value} bpm"
        else:
            # line = f"{label}: {data.hex()}"
            line = f"{label}"

        output.append(line)

    if output:
        return "\n".join(output)
    else:
        return "—"


class RSSIKalman:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # initial state
        self.kf.x = np.array([[-59.0], [0.0]], dtype=np.float64)

        # state transition
        self.kf.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        # measurement function
        self.kf.H = np.array([[1.0, 0.0]], dtype=np.float64).reshape((1, 2))

        # initial uncertainty
        self.kf.P *= 10.0

        # measurement noise
        # self.kf.R = np.array([[2.0]], dtype=np.float64)
        self.kf.R = np.array(
            [[10.0]], dtype=np.float64
        )  # more smoothing, trust rssi less

        # process noise
        # self.kf.Q = np.array([[1e-4, 0.0], [0.0, 1e-4]], dtype=np.float64)

        # self.kf.Q = np.array(
        #     [[1e-5, 0.0], [0.0, 1e-5]], dtype=np.float64
        # )  # less process noise

        self.kf.Q = np.array(
            [[1e-4, 0.0], [0.0, 5e-4]], dtype=np.float64
        )  # Smooth but allows real trend

    def update(self, rssi: float) -> float:
        self.kf.predict()
        self.kf.update(np.array([[rssi]], dtype=np.float64))
        return self.kf.x[0, 0]


rssi_filters = defaultdict(RSSIKalman)

# address -> dict(name, rssi, last_seen)
devices = {}


def detection_callback(device, advertisement_data):
    devices[device.address] = {
        "name": device.name or "Unknown",
        "rssi": advertisement_data.rssi,
        "tx_power": advertisement_data.tx_power,
        "manufacturer_data": advertisement_data.manufacturer_data,
        "service_data": advertisement_data.service_data,
        "service_uuids": advertisement_data.service_uuids,
        "platform_data": advertisement_data.platform_data,
        "last_seen": time.monotonic(),
    }


async def scan_loop():
    # adapter = get_adapter_hci_by_mac(PREFERRED_ADAPTER_MAC)
    # if not adapter:
    #     console.print(
    #         f"[bold red]Could not find adapter with MAC {PREFERRED_ADAPTER_MAC}"
    #     )
    #     return

    # scanner = BleakScanner(detection_callback, adapter=adapter)

    scanner = BleakScanner(detection_callback)
    await scanner.start()

    try:
        with Live(refresh_per_second=4) as live:  # Live table rendering
            while True:
                await asyncio.sleep(1.0)

                table = Table(title="")
                table.add_column("Name", style="cyan", no_wrap=True)
                table.add_column("Address", style="magenta")
                table.add_column("Raw", justify="right")
                # table.add_column("Smoothed RSSI", justify="right")
                table.add_column("K.F RSSI (dBm)", justify="right")
                table.add_column("Distance (m)", justify="right")
                table.add_column("Age (s)", justify="right")
                table.add_column("Tx Power", justify="right")
                table.add_column("Manufacturer Data", style="yellow")
                table.add_column("Service Data UUID", style="blue")
                # table.add_column("Last Seen (s ago)", justify="right")
                table.add_column(
                    "Service UUIDs", style="cyan", no_wrap=True, justify="right"
                )

                rows = []
                now = time.monotonic()

                for address, info in list(devices.items()):
                    age = now - info["last_seen"]
                    if age > DEVICE_TIMEOUT:
                        continue

                    raw_rssi = info["rssi"]
                    clamped_rssi = max(min(raw_rssi, -30), -100)
                    smoothed_rssi = rssi_filters[address].update(clamped_rssi)
                    distance = estimate_distance(smoothed_rssi)

                    if distance and distance <= RANGE:
                        manuf_data = decode_manufacturer_data(
                            info.get("manufacturer_data", {})
                        )
                        service_data = decode_service_data(info.get("service_data", {}))
                        row = (
                            distance if distance else float("inf"),
                            [
                                info["name"],
                                address,
                                f"{raw_rssi}",
                                f"{smoothed_rssi:.0f}",
                                (
                                    f"[green]{distance:.1f}[/green]"
                                    if distance and distance <= WARN
                                    else f"{distance:.1f}" if distance else "N/A"
                                ),
                                f"{age:.0f}",
                                f"{info.get('tx_power', 'N/A')}",
                                manuf_data,
                                service_data,
                                str(len(info.get("service_uuids"))),
                            ],
                        )
                        rows.append(row)

                rows.sort(key=lambda x: x[0])

                for _, row in rows:
                    table.add_row(*row)

                live.update(table)

    finally:
        await scanner.stop()


if __name__ == "__main__":
    try:
        console.clear()
        asyncio.run(scan_loop())
    except KeyboardInterrupt:
        console.print("\n[bold red]Scan stopped by user.")
