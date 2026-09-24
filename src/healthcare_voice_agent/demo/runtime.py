"""Minimal session state: identity, turn boundary and in-flight transactions.

No caller text parsing, conversation phases or audio-delivery receipts.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from healthcare_voice_agent.booking.service import BookingService
from healthcare_voice_agent.demo.database import (
    DemoDatabaseError, build_demo_engine, demo_case_summary, verify_demo_database,
)


@dataclass
class DemoRuntime:
    engine: object
    service: BookingService
    context: object
    summary: dict
    trace: object = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    epoch: int = 0
    final_turn_epoch: int = -1
    closed: bool = False
    pending: dict | None = None
    results: dict = field(default_factory=dict)
    last_result: dict | None = None

    def emit(self, event, **data):
        if self.trace is not None:
            try:
                self.trace.emit(event, **data)
            except Exception:
                pass  # Logging must not change a transaction outcome.

    async def db(self, method, *args, **kwargs):
        """Finish an in-flight DB call even if its awaiting audio turn is cancelled.

        Callers hold lock until they have stored the result. Disconnect cleanup
        waits for that same lock, so it cannot dispose an active transaction.
        """
        task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()

    def interrupt(self):
        self.epoch += 1  # Audio generation only; NOT a business conversation turn.

    async def user_turn(self):
        async with self.lock:
            if self.closed:
                return
            self.interrupt()
            turn_epoch = self.epoch
            self.context = await self.db(self.service.advance_conversation, self.context)
            self.results.clear()
            self.final_turn_epoch = turn_epoch
            return turn_epoch

    async def close(self):
        async with self.lock:
            if self.closed:
                return
            self.closed = True
            self.interrupt()
            try:
                await self.db(self.service.close_session, self.context)
            finally:
                await self.db(self.engine.dispose)


def _open_demo(trace):
    engine = build_demo_engine()
    service, context = BookingService(engine), None
    try:
        verify_demo_database(engine)
        summary = demo_case_summary(engine)
        context = service.open_session(clinic_id=summary["clinic_id"], user_id=summary["staff_id"])
        context = service.set_confirmed_case(context, summary["case_id"])
        return DemoRuntime(engine, service, context, summary, trace)
    except Exception:
        if context is not None:
            try:
                service.close_session(context)
            except Exception:
                pass
        engine.dispose()
        raise DemoDatabaseError("The isolated demo session could not be opened. Run the demo readiness check.") from None


async def open_demo_runtime(trace):
    task = asyncio.create_task(asyncio.to_thread(_open_demo, trace))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        if not task.cancelled() and task.exception() is None:
            cleanup = asyncio.create_task(task.result().close())
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
        raise
