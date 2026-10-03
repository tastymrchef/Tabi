"""
Block C -- Postgres connection setup for the durable checkpointer and the
long-term Store.

Real Windows-specific problem: AsyncPostgresSaver and AsyncPostgresStore
both need an asyncio event loop that supports psycopg's async connection
style (a "selector" loop). But places.py's MCP tool needs Windows'
subprocess-capable loop (a "proactor" loop) to launch `npx`. Windows
can't give one loop both things at once -- confirmed, not just us:
https://github.com/bytedance/deer-flow/issues/5936 hit the identical
psycopg-async vs. stdio-MCP-subprocess conflict.

The fix (same one used for the checkpointer): run both Postgres
connections on one dedicated background thread with its own
selector-based event loop. The main thread's proactor loop never touches
Postgres directly -- it hands operations off to this thread and gets the
result back, so it stays free to keep launching MCP subprocesses.
"""

import asyncio
import os
import threading

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def get_db_uri() -> str:
    db_uri = os.environ.get("DATABASE_URL")
    if not db_uri:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file with the "
            "connection string Neon gave you."
        )
    return db_uri


class _LoopThread:
    """Owns a selector-based event loop running forever on its own
    background thread. `run(coro)` submits a coroutine to that loop from
    any other thread and awaits the result."""

    def __init__(self):
        self._loop = None
        self._thread = None

    def start(self):
        ready = threading.Event()

        def _target():
            self._loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()
        ready.wait()

    async def stop(self):
        async def _drain():
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        await self.run(_drain())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    async def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)


class _CheckpointerProxy(BaseCheckpointSaver):
    """Satisfies LangGraph's isinstance(checkpointer, BaseCheckpointSaver)
    check, while actually running every operation on the Postgres
    thread's loop instead of the caller's."""

    def __init__(self, target, bridge):
        super().__init__()
        self._target = target
        self._bridge = bridge

    async def aget_tuple(self, config):
        return await self._bridge.run(self._target.aget_tuple(config))

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._bridge.run(
            self._target.aput(config, checkpoint, metadata, new_versions)
        )

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await self._bridge.run(
            self._target.aput_writes(config, writes, task_id, task_path)
        )

    async def alist(self, config, *, filter=None, before=None, limit=None):
        agen = self._target.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield await self._bridge.run(agen.__anext__())
            except StopAsyncIteration:
                break

    async def adelete_thread(self, thread_id):
        return await self._bridge.run(self._target.adelete_thread(thread_id))

    def get_next_version(self, current, channel):
        return self._target.get_next_version(current, channel)

    async def setup(self):
        return await self._bridge.run(self._target.setup())


class _StoreProxy(BaseStore):
    """Same idea as _CheckpointerProxy, but for the Store. BaseStore
    builds aget/aput/adelete/asearch/alist_namespaces on top of a single
    abatch() call internally, so bridging just that one method is enough
    to make all of them safe."""

    def __init__(self, target, bridge):
        super().__init__()
        self._target = target
        self._bridge = bridge

    async def abatch(self, ops):
        return await self._bridge.run(self._target.abatch(ops))

    def batch(self, ops):
        raise NotImplementedError("This store only supports async access.")

    async def setup(self):
        return await self._bridge.run(self._target.setup())


class postgres_backend:
    """Async context manager giving you both a checkpointer and a
    long-term store, backed by two Postgres connections sharing one
    dedicated background thread:

        async with postgres_backend(get_db_uri()) as (checkpointer, store):
            await checkpointer.setup()
            await store.setup()
            ...
    """

    def __init__(self, conn_string: str):
        self._conn_string = conn_string
        self._bridge = _LoopThread()
        self._pool = None

    async def __aenter__(self):
        self._bridge.start()

        async def _open():
            pool = AsyncConnectionPool(
                self._conn_string,
                min_size=1,
                max_size=5,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
                # Validate (and silently replace) a connection before handing
                # it out, instead of discovering it's dead mid-query. This is
                # what actually fixes "SSL connection has been closed
                # unexpectedly" -- Neon drops idle connections, and holding a
                # single connection open for the bot's entire runtime will
                # eventually hit one of those drops.
                check=AsyncConnectionPool.check_connection,
                # Recycle idle/old connections proactively too, rather than
                # relying on check_connection to catch every case.
                max_idle=300,
                max_lifetime=1800,
                open=False,
            )
            await pool.open()
            self._pool = pool

            raw_checkpointer = AsyncPostgresSaver(conn=pool)
            raw_store = AsyncPostgresStore(conn=pool)
            return raw_checkpointer, raw_store

        raw_checkpointer, raw_store = await self._bridge.run(_open())
        checkpointer = _CheckpointerProxy(raw_checkpointer, self._bridge)
        store = _StoreProxy(raw_store, self._bridge)
        return checkpointer, store

    async def __aexit__(self, exc_type, exc, tb):
        async def _close():
            await self._pool.close()

        await self._bridge.run(_close())
        await self._bridge.stop()