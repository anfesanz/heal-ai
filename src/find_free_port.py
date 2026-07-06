"""Find an available local TCP port."""

from __future__ import annotations

import argparse
import socket


def find_free_port(start: int, attempts: int = 100) -> int:
    """Return the first available localhost port at or above `start`."""

    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found from {start} to {start + attempts - 1}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find a free local port.")
    parser.add_argument("--start", type=int, default=8501)
    parser.add_argument("--attempts", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(find_free_port(args.start, args.attempts))
