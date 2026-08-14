from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import serial

from at2410_protocol import (
    AT2410StreamParser,
    TargetInfo,
    parse_frame,
    run_protocol_self_test,
)


BOARD_DEFAULT_BAUDRATE = 9600
AT2410_PORT_ENV = "MMWAVE_AT2410_PORT"
BOARD_DEFAULT_PORT_CANDIDATES = (
    "/dev/serial/by-id/usb-SIPEED_UARTx4_HS_FactoryAIOT_Prog_Serial-if00",
    "/dev/ttyACM0",
)


def default_serial_port() -> str:
    configured = os.environ.get(AT2410_PORT_ENV, "").strip()
    if configured:
        return configured
    for port in BOARD_DEFAULT_PORT_CANDIDATES:
        if os.path.exists(port):
            return port
    return BOARD_DEFAULT_PORT_CANDIDATES[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AT2410 UART test tool")
    parser.add_argument(
        "--port",
        default=default_serial_port(),
        help=f"serial port path (default from {AT2410_PORT_ENV}, otherwise SIPEED if00/ttyACM0)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=BOARD_DEFAULT_BAUDRATE,
        help=f"serial baudrate (current board default: {BOARD_DEFAULT_BAUDRATE})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="capture duration in seconds; 0 means listen until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=256,
        help="bytes to read per iteration",
    )
    parser.add_argument(
        "--undetected-interval",
        type=float,
        default=1.0,
        help="seconds between Undetected messages when no frame is parsed",
    )
    parser.add_argument(
        "--range-unit",
        choices=("mm", "cm"),
        default="cm",
        help="distance unit reported by radar payload",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print debug information such as open/done/warning",
    )
    parser.add_argument(
        "--raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print every raw read in hex, including <EMPTY> for zero bytes (default: enabled)",
    )
    parser.add_argument(
        "--flush-input",
        action="store_true",
        help="discard bytes already queued when the port is opened",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify the documented 1/2/3-target frames without opening a serial port",
    )
    parser.add_argument(
        "--decode-hex",
        metavar="HEX",
        help="decode one complete hexadecimal AT2410 frame without opening a serial port",
    )
    return parser.parse_args()


def format_range_m(distance_raw: int, range_unit: str) -> float:
    if range_unit == "cm":
        return distance_raw / 100.0
    return distance_raw / 1000.0


def is_valid_target(target: TargetInfo) -> bool:
    return 1 <= target.target_id <= 8 and target.distance_cm > 0


def print_line(message: str, *, error: bool = False) -> None:
    print(message, file=sys.stderr if error else sys.stdout, flush=True)


def available_serial_ports() -> list[str]:
    patterns = ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyS*")
    return sorted({port for pattern in patterns for port in glob.glob(pattern)})


def print_target(target: TargetInfo, *, index: int, count: int, range_unit: str) -> None:
    plausible = is_valid_target(target)
    suffix = "" if plausible else " [warning: ID/range outside documented values]"
    if target.speed_cm_s < 0:
        motion = "approaching"
    elif target.speed_cm_s > 0:
        motion = "receding"
    else:
        motion = "stationary"
    timestamp = time.strftime("%H:%M:%S")
    print_line(
        f"{timestamp}: Target {index}/{count}, ID={target.target_id}, "
        f"Angle={target.angle_deg} deg, "
        f"Range={format_range_m(target.distance_cm, range_unit):.2f} m "
        f"({target.distance_cm} cm), "
        f"Velocity={target.speed_cm_s / 100.0:.2f} m/s "
        f"({target.speed_cm_s} cm/s), Motion={motion}{suffix}"
    )


def print_frame(frame, *, frame_count: int, range_unit: str, verbose: bool) -> None:
    if verbose:
        print_line(
            f"frame={frame_count} type=0x{frame.command:02X} "
            f"targets={frame.object_count} LEN=0x{frame.raw_frame[1]:02X} "
            f"checksum=0x{frame.checksum:02X} raw={frame.raw_frame.hex(' ').upper()}"
        )
    for index, target in enumerate(frame.targets, start=1):
        print_target(target, index=index, count=frame.object_count, range_unit=range_unit)


def run_self_test() -> int:
    try:
        lines = run_protocol_self_test()
    except (AssertionError, ValueError) as exc:
        print_line(f"self-test failed: {exc}", error=True)
        return 1
    for line in lines:
        print_line(f"self-test: {line}")
    print_line("self-test: PASS")
    return 0


def decode_hex_frame(value: str, range_unit: str) -> int:
    try:
        raw_frame = bytes.fromhex(value)
        frame = parse_frame(raw_frame)
    except ValueError as exc:
        print_line(f"decode failed: {exc}", error=True)
        return 1
    print_frame(frame, frame_count=1, range_unit=range_unit, verbose=True)
    return 0


def open_serial_port(port: str, baudrate: int) -> serial.Serial:
    # Configure modem-control lines before opening the CDC ACM device. Opening
    # it through Serial(port=...) briefly asserts DTR/RTS, which stops RX on
    # the SIPEED UARTx4 bridge used by this board.
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 0.2
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def main() -> int:
    args = parse_args()

    if args.self_test:
        return run_self_test()
    if args.decode_hex is not None:
        return decode_hex_frame(args.decode_hex, args.range_unit)

    parser = AT2410StreamParser()
    total_bytes = 0
    frame_count = 0
    target_count = 0
    last_output_at = time.monotonic()

    if args.verbose:
        duration_text = "continuous" if args.duration <= 0 else f"{args.duration:.1f}s"
        print_line(
            f"open port={args.port} baudrate={args.baudrate} duration={duration_text}; "
            "prefixes=5A-0D-0A-01/5A-15-0A-02/5A-1D-0A-03"
        )
    if not os.path.exists(args.port):
        available = ", ".join(available_serial_ports()) or "none"
        print_line(
            f"error: serial port does not exist: {args.port}; available ports: {available}",
            error=True,
        )
        return 4

    try:
        with open_serial_port(args.port, args.baudrate) as ser:
            # The radar may only report while a target is present. Do not throw
            # away an already queued report unless the caller explicitly asks.
            if args.flush_input:
                ser.reset_input_buffer()
            deadline = None if args.duration <= 0 else time.monotonic() + args.duration

            while deadline is None or time.monotonic() < deadline:
                chunk = ser.read(args.read_size)
                now = time.monotonic()
                if not chunk:
                    if now - last_output_at >= args.undetected_interval:
                        timestamp = time.strftime("%H:%M:%S")
                        if args.raw:
                            print_line(
                                f"{timestamp}: RAW RX [0 bytes]: <EMPTY> "
                                f"total_bytes={total_bytes} frames={frame_count} "
                                f"targets={target_count} pending={parser.pending_bytes}"
                            )
                        last_output_at = now
                    continue

                total_bytes += len(chunk)
                if args.raw:
                    timestamp = time.strftime("%H:%M:%S")
                    print_line(
                        f"{timestamp}: RAW RX [{len(chunk)} bytes]: "
                        f"{chunk.hex(' ').upper()}"
                    )
                frames = parser.feed(chunk)
                if not frames and now - last_output_at >= args.undetected_interval:
                    timestamp = time.strftime("%H:%M:%S")
                    print_line(
                        f"{timestamp}: Receiving data, waiting for a valid complete frame "
                        f"(bytes={total_bytes}, header_errors={parser.stats.header_errors}, "
                        f"checksum_errors={parser.stats.checksum_errors}, "
                        f"pending={parser.pending_bytes})"
                    )
                    last_output_at = now

                for frame in frames:
                    frame_count += 1
                    target_count += len(frame.targets)
                    last_output_at = time.monotonic()
                    print_frame(
                        frame,
                        frame_count=frame_count,
                        range_unit=args.range_unit,
                        verbose=args.verbose,
                    )
    except KeyboardInterrupt:
        print_line(
            f"stopped: bytes={total_bytes}, frames={frame_count}, targets={target_count}"
        )
        return 130
    except (OSError, serial.SerialException) as exc:
        print_line(f"error: failed to use serial port {args.port}: {exc}", error=True)
        return 4

    if total_bytes == 0:
        timestamp = time.strftime("%H:%M:%S")
        print_line(
            f"{timestamp}: no serial bytes received from {args.port}. This can mean "
            "there was no target (the protocol sends no frame in that case), or the "
            "port mapping/power/TX wiring/baudrate is wrong",
            error=True,
        )
        return 2
    if frame_count == 0:
        timestamp = time.strftime("%H:%M:%S")
        print_line(
            f"warning: received {total_bytes} bytes from {args.port}, but no valid "
            f"AT2410 frame was parsed; header_errors={parser.stats.header_errors}, "
            f"checksum_errors={parser.stats.checksum_errors}, "
            f"discarded={parser.stats.bytes_discarded}, pending={parser.pending_bytes}, "
            f"last_error={parser.stats.last_error or 'none'}; check baudrate and UART channel",
            error=True,
        )
        return 3
    if args.verbose:
        print_line(
            f"done: bytes={total_bytes}, frames={frame_count}, targets={target_count}, "
            f"header_errors={parser.stats.header_errors}, "
            f"checksum_errors={parser.stats.checksum_errors}, "
            f"discarded={parser.stats.bytes_discarded}, pending={parser.pending_bytes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
