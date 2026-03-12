"""Python 3.14+ compatibility for ib_insync / eventkit."""
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
