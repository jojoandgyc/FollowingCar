"""Command line helpers for the LZ-30EMA RS485 client."""

from __future__ import annotations

import argparse
import time

from .client import LZ30EMAClient
from .status import format_realtime_snapshot, protocol_field_coverage_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Print LZ-30EMA_2EC_N realtime RS485 status.")
    parser.add_argument("--port", help="Serial port, for example COM3 or /dev/ttyUSB0.")
    parser.add_argument("--slave", type=int, default=1, help="Modbus slave address. Default: 1.")
    parser.add_argument("--baudrate", type=int, default=9600, help="Serial baudrate. Default: 9600.")
    parser.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout in seconds.")
    parser.add_argument("--parity", default="N", choices=("N", "E", "O"), help="Serial parity.")
    parser.add_argument("--stopbits", type=int, default=1, choices=(1, 2), help="Serial stop bits.")
    parser.add_argument(
        "--rs485-mode",
        default="auto",
        choices=("auto", "none", "rts-high", "rts-low"),
        help="Linux RS485 direction mode. Default: auto.",
    )
    parser.add_argument(
        "--only-supported",
        action="store_true",
        help="Print only fields that have RS485 realtime registers.",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Print which screenshot fields are/are not available in the protocol.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep reading until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=1.0, help="Watch interval in seconds. Default: 1.")
    parser.add_argument("--count", type=int, help="Stop after this many watch reads.")
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="In watch mode, print only left/right speed and errors.",
    )
    args = parser.parse_args()

    if args.coverage:
        print(protocol_field_coverage_text())
        if not args.port:
            return 0

    if not args.port:
        parser.error("--port is required unless --coverage is used")

    client = LZ30EMAClient.from_serial(
        args.port,
        slave=args.slave,
        baudrate=args.baudrate,
        timeout=args.timeout,
        parity=args.parity,
        stopbits=args.stopbits,
        rs485_mode=args.rs485_mode,
    )
    try:
        if args.watch:
            reads = 0
            while True:
                reads += 1
                if args.speed_only:
                    left = client.read_motor_status("left")
                    right = client.read_motor_status("right")
                    print(
                        f"{time.strftime('%H:%M:%S')} "
                        f"left={left.speed_rpm:>7} RPM ({left.error_name})  "
                        f"right={right.speed_rpm:>7} RPM ({right.error_name})",
                        flush=True,
                    )
                else:
                    snapshot = client.read_realtime_snapshot()
                    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
                    print(format_realtime_snapshot(snapshot, include_unsupported=not args.only_supported))
                if args.count is not None and reads >= args.count:
                    break
                time.sleep(max(args.interval, 0.05))
        else:
            snapshot = client.read_realtime_snapshot()
            print(format_realtime_snapshot(snapshot, include_unsupported=not args.only_supported))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
