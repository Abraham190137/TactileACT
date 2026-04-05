"""
TCP client helper for TactileACT deployment.

Matches the framing protocol in ws_server.py:
  [4-byte big-endian length][pickle payload]
"""
from __future__ import annotations

import pickle
import socket
import struct
import time
import logging


def _send_msg(sock: socket.socket, data: bytes) -> None:
    header = struct.pack(">I", len(data))
    sock.sendall(header + data)


def _recv_msg(sock: socket.socket) -> bytes | None:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    length = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(length - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return data


def pack(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def unpack(data: bytes):
    return pickle.loads(data)


def connect_to_server(host: str, port: int, retry_interval: float = 3.0) -> socket.socket:
    """Connect to server with retry. Returns connected socket."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            logging.info("[client] Connected to %s:%d", host, port)
            return sock
        except ConnectionRefusedError:
            logging.info("[client] Server not ready, retrying in %.0fs...", retry_interval)
            time.sleep(retry_interval)


def recv_metadata(sock: socket.socket) -> dict:
    """Receive the initial metadata frame from server."""
    raw = _recv_msg(sock)
    if raw is None:
        raise ConnectionError("Server closed before sending metadata")
    return unpack(raw)


def send_obs(sock: socket.socket, obs: dict) -> None:
    """Send observation dict to server."""
    _send_msg(sock, pack(obs))


def recv_action(sock: socket.socket) -> dict:
    """Receive action dict from server."""
    raw = _recv_msg(sock)
    if raw is None:
        raise ConnectionError("Server closed connection")
    return unpack(raw)
