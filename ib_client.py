"""IBKR Gateway connection manager for the premarket system."""
import asyncio
import logging
import threading
from typing import Optional

import ib_compat  # noqa: F401 — must be imported before ib_insync (Python 3.14+ event loop fix)
from ib_insync import IB

import config

logger = logging.getLogger(__name__)


def _ensure_event_loop():
    """Ensure the current thread has an event loop (required by ib_insync on Python 3.14+)."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


class IBClient:
    """
    Lightweight IB connection wrapper.
    Connects once from whichever thread first needs it,
    then reuses that connection.
    """

    _instance: Optional["IBClient"] = None
    _ib: Optional[IB] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def ib(self) -> IB:
        _ensure_event_loop()
        with self._lock:
            if self._ib is None or not self._ib.isConnected():
                self._connect()
            return self._ib

    def _connect(self):
        """Connect to IB Gateway."""
        _ensure_event_loop()
        self._ib = IB()
        self._ib.connect(
            host=config.IB_HOST,
            port=config.IB_PORT,
            clientId=config.IB_CLIENT_ID,
            timeout=20,
            readonly=True,
        )
        logger.info(
            "Connected to IB Gateway at %s:%s (clientId=%s)",
            config.IB_HOST, config.IB_PORT, config.IB_CLIENT_ID,
        )

    def disconnect(self):
        with self._lock:
            if self._ib and self._ib.isConnected():
                self._ib.disconnect()
                logger.info("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()


# Global singleton
ib_client = IBClient()
