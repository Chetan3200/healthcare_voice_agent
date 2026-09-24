"""Offline one-shot artificial clinician web-search delay checks."""
import asyncio

from healthcare_voice_agent.clinician.runtime import ClinicianRuntime


SEARCH_ARGS = {
    "query": "general wrist fracture guidance",
    "top_k": 2,
    "document_id": None,
    "version": None,
}


class Guidance:
    def __init__(self):
        self.calls = []

    def search(self, args):
        self.calls.append(args.query)
        return {"query": args.query, "version_scope": "web_unversioned", "provider": "exa",
                "matches": [], "as_of": "2026-01-01T00:00:00+00:00"}, []


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class BlockingSleep:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)
        self.entered.set()
        await self.release.wait()


def test_first_valid_search_claims_delay_while_second_search_proceeds():
    async def check():
        guidance, trace, sleep = Guidance(), Trace(), BlockingSleep()
        runtime = ClinicianRuntime(None, guidance, None, None, trace,
                                  demo_web_search_delay_seconds=2.5, sleep=sleep)
        first = asyncio.create_task(runtime.invoke("search_clinic_instructions", SEARCH_ARGS))
        await sleep.entered.wait()
        second = await runtime.invoke("search_clinic_instructions", SEARCH_ARGS)
        assert second["status"] == "ok"
        assert guidance.calls == [SEARCH_ARGS["query"]]
        assert sleep.calls == [2.5]
        delay_events = [fields for event, fields in trace.events
                        if event == "clinician_artificial_web_search_delay"]
        assert delay_events == [{"tool_name": "search_clinic_instructions",
                                 "artificial_delay_seconds": 2.5}]
        sleep.release.set()
        first_result = await first
        assert first_result["status"] == "ok"
        assert guidance.calls == [SEARCH_ARGS["query"], SEARCH_ARGS["query"]]

    asyncio.run(check())


def test_disabled_invalid_and_closed_searches_do_not_sleep_or_claim_invalid_closed():
    async def check():
        guidance, trace = Guidance(), Trace()

        async def unexpected_sleep(_):
            raise AssertionError("disabled delay must not sleep")

        runtime = ClinicianRuntime(None, guidance, None, None, trace,
                                  demo_web_search_delay_seconds=0.0, sleep=unexpected_sleep)
        assert (await runtime.invoke("search_clinic_instructions", SEARCH_ARGS))["status"] == "ok"
        assert len([event for event, _ in trace.events
                    if event == "clinician_artificial_web_search_delay"]) == 1

        invalid_trace = Trace()
        invalid_runtime = ClinicianRuntime(None, Guidance(), None, None, invalid_trace,
                                          demo_web_search_delay_seconds=3, sleep=unexpected_sleep)
        invalid = await invalid_runtime.invoke("search_clinic_instructions", {"query": "x"})
        assert invalid["error"]["code"] == "INVALID_ARGUMENT"
        assert not [event for event, _ in invalid_trace.events
                    if event == "clinician_artificial_web_search_delay"]

        closed_trace = Trace()
        closed_guidance = Guidance()
        closed_runtime = ClinicianRuntime(None, closed_guidance, None, None, closed_trace,
                                          demo_web_search_delay_seconds=3, sleep=unexpected_sleep)
        await closed_runtime.close()
        closed = await closed_runtime.invoke("search_clinic_instructions", SEARCH_ARGS)
        assert closed["error"]["code"] == "BACKEND_UNAVAILABLE"
        assert not closed_guidance.calls
        assert not [event for event, _ in closed_trace.events
                    if event == "clinician_artificial_web_search_delay"]

    asyncio.run(check())


def test_closing_during_delay_prevents_the_real_search():
    async def check():
        guidance, sleep = Guidance(), BlockingSleep()
        runtime = ClinicianRuntime(None, guidance, None, None,
                                  demo_web_search_delay_seconds=1, sleep=sleep)
        search = asyncio.create_task(runtime.invoke("search_clinic_instructions", SEARCH_ARGS))
        await sleep.entered.wait()
        await runtime.close()
        sleep.release.set()
        result = await search
        assert result["error"]["code"] == "BACKEND_UNAVAILABLE"
        assert not guidance.calls

    asyncio.run(check())
