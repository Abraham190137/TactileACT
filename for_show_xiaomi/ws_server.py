"""
TCP socket server for TactileACT deployment.

Simple framing protocol over raw TCP:
  - Each message is: [4-byte big-endian length][payload bytes]
  - Payload is pickle-serialized dict (with numpy arrays)

Protocol flow:
  1. Client connects
  2. Server sends metadata dict
  3. Loop: client sends obs dict → server sends action dict
"""
from __future__ import annotations

import logging
import pickle
import queue
import socket
import struct
import threading
from typing import Any

import numpy as np


class ClientDisconnected(RuntimeError):
    pass


def _send_msg(sock: socket.socket, data: bytes) -> None:
    """Send a length-prefixed message."""
    header = struct.pack(">I", len(data))
    sock.sendall(header + data)


def _recv_msg(sock: socket.socket) -> bytes | None:
    """Receive a length-prefixed message. Returns None on disconnect."""
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


def _pack(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _unpack(data: bytes):
    return pickle.loads(data)


_DISCONNECT_SENTINEL = object()


class TactileACTServer:
    """Blocking single-client TCP server for action serving."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        metadata: dict[str, Any] | None = None,
    ):
        self._host = host
        self._port = int(port)
        self._metadata: dict[str, Any] = metadata or {
            "protocol": "tactileact",
        }

        self._req_q: queue.Queue = queue.Queue()
        self._resp_q: queue.Queue = queue.Queue()

        self._state_lock = threading.Lock()
        self._next_token = 0
        self._active_token: int | None = None
        self._last_req_token: int | None = None

        self._server_socket: socket.socket | None = None
        self._serve_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._server_socket is not None:
            raise RuntimeError("Server already started")

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)  # for clean shutdown
        self._running = True

        self._serve_thread = threading.Thread(
            target=self._accept_loop,
            name=f"tactileact-server-{self._host}:{self._port}",
            daemon=True,
        )
        self._serve_thread.start()
        logging.info("[server] Listening on %s:%d", self._host, self._port)

    def close(self) -> None:
        self._running = False
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=3.0)
            self._serve_thread = None

    def recv_obs(self) -> dict[str, Any]:
        token, msg = self._req_q.get()
        if msg is _DISCONNECT_SENTINEL:
            raise ClientDisconnected("Client disconnected")
        if not isinstance(msg, dict):
            raise TypeError(f"Expected obs dict, got {type(msg)}")
        with self._state_lock:
            self._last_req_token = int(token)
        return msg

    def send_action(self, action_dict: dict) -> None:
        with self._state_lock:
            token = self._last_req_token
            active = self._active_token
        if token is None or active is None or token != active:
            raise ClientDisconnected("No active client for send_action")
        self._resp_q.put((int(token), action_dict))

    def _drain_queue(self, q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            logging.info("[server] Client connected: %s", addr)
            t = threading.Thread(
                target=self._handle_client,
                args=(conn, addr),
                daemon=True,
            )
            t.start()

    def _handle_client(self, conn: socket.socket, addr) -> None:
        with self._state_lock:
            self._next_token += 1
            my_token = int(self._next_token)
            self._active_token = my_token
            self._last_req_token = None

        self._drain_queue(self._req_q)
        self._drain_queue(self._resp_q)

        try:
            # Send metadata
            _send_msg(conn, _pack(self._metadata))

            while self._running:
                raw = _recv_msg(conn)
                if raw is None:
                    break
                req = _unpack(raw)
                if not isinstance(req, dict):
                    raise TypeError(f"Expected dict, got {type(req)}")
                self._req_q.put((my_token, req))

                # Wait for response
                while True:
                    resp_token, resp = self._resp_q.get()
                    if int(resp_token) == my_token:
                        _send_msg(conn, _pack(resp))
                        break
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            logging.info("[server] Client disconnected: %s", addr)
            with self._state_lock:
                if self._active_token == my_token:
                    self._active_token = None
            self._req_q.put((my_token, _DISCONNECT_SENTINEL))
            try:
                conn.close()
            except Exception:
                pass
