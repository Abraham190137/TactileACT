"""
Lightweight serializer for dicts containing numpy arrays.

Uses pickle internally (both ends are Python), wrapped to match
the Packer/packb/unpackb interface used by bushu_lizi ws_server.
"""
from __future__ import annotations

import pickle

import numpy as np


class Packer:
    def pack(self, obj) -> bytes:
        return packb(obj)


def packb(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def unpackb(data: bytes):
    return pickle.loads(data)
