from __future__ import annotations

import logging
import queue
import threading
import traceback
from typing import Any

import numpy as np
import websockets.exceptions
import websockets.frames
import websockets.sync.server

from . import msgpack_numpy


class ClientDisconnected(RuntimeError):
    pass


_DISCONNECT_SENTINEL = object()


class MIACTWebsocketActionServer:
    """Blocking single-client websocket bridge for action serving.

    MITA-VLA-aligned framing:
      - server sends one metadata frame after websocket connect
      - client sends plain obs dict
      - server sends action dict

    For MIACT single-step control, action dict is encoded as H=1 chunk:
      {"actions": np.float32[1, action_dim]}
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        metadata: dict[str, Any] | None = None,
    ):
        self._host = host
        self._port = int(port)
        self._packer = msgpack_numpy.Packer()
        self._metadata: dict[str, Any] = metadata or {
            "protocol": "mitavla_compatible",
            "action_horizon": 1,
        }

        self._req_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._resp_q: queue.Queue[dict[str, Any]] = queue.Queue()

        self._state_lock = threading.Lock()
        self._next_token = 0
        self._active_token: int | None = None
        self._last_req_token: int | None = None

        self._server: websockets.sync.server.Server | None = None
        self._serve_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Websocket server already started")

        self._server = websockets.sync.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        )
        # NOTE: websockets.sync.server.serve() returns a Server context manager.
        # Entering the context binds/listens; serve_forever() actually accepts and
        # processes connections, so we run it in a background thread.
        self._server.__enter__()
        self._serve_thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"miact-ws-serve-{self._host}:{self._port}",
            daemon=True,
        )
        self._serve_thread.start()
        logging.info("[miact-ws-server] listening on ws://%s:%d", self._host, self._port)

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            finally:
                self._server.__exit__(None, None, None)
                self._server = None

        if self._serve_thread is not None:
            self._serve_thread.join(timeout=2.0)
            self._serve_thread = None

    def recv_obs(self) -> dict[str, Any]:
        token, msg = self._req_q.get()
        if msg is _DISCONNECT_SENTINEL:
            raise ClientDisconnected("Websocket client disconnected")
        if not isinstance(msg, dict):
            raise TypeError(f"Expected obs dict from client, got {type(msg)}")

        with self._state_lock:
            self._last_req_token = int(token)
        return msg

    def send_action(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        with self._state_lock:
            token = self._last_req_token
            active = self._active_token

        # If the client disconnected after providing obs, do NOT enqueue a stale
        # action into the shared queue; treat it as disconnect so the policy loop
        # can restart from a clean episode on next connect.
        if token is None or active is None or token != active:
            raise ClientDisconnected("No active websocket client for send_action")

        self._resp_q.put((int(token), {"actions": a[None, :]}))

    def _drain_queue(self, q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def _handler(self, conn: websockets.sync.server.ServerConnection) -> None:
        try:
            remote = conn.remote_address
        except Exception:
            remote = "<unknown>"

        with self._state_lock:
            self._next_token += 1
            my_token = int(self._next_token)
            self._active_token = my_token
            self._last_req_token = None

        # Clear any leftovers from a previous session, so a new client always
        # starts clean (mitavla-like behavior).
        self._drain_queue(self._req_q)
        self._drain_queue(self._resp_q)

        logging.info("[miact-ws-server] client connected: %s (token=%s)", remote, my_token)
        conn.send(self._packer.pack(self._metadata))

        while True:
            try:
                raw = conn.recv()
                req = msgpack_numpy.unpackb(raw)
                if not isinstance(req, dict):
                    raise TypeError(f"Expected dict request, got {type(req)}")
                self._req_q.put((my_token, req))

                # Only send responses that belong to this connection; discard
                # anything stale from a previous token.
                while True:
                    resp_token, resp = self._resp_q.get()
                    if int(resp_token) == my_token:
                        conn.send(self._packer.pack(resp))
                        break
            except websockets.exceptions.ConnectionClosed:
                logging.info(
                    "[miact-ws-server] client disconnected: %s",
                    remote,
                )
                with self._state_lock:
                    if self._active_token == my_token:
                        self._active_token = None
                # Unblock the policy loop if it's waiting for obs.
                self._req_q.put((my_token, _DISCONNECT_SENTINEL))
                break
            except Exception:
                conn.send(traceback.format_exc())
                conn.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
