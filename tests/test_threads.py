"""The LLM client, HTTP client and cache are shared by parallel graph branches (task A12)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx

from mia import cache, http, runtime
from mia.llm import LLM, LLMBudgetExceeded
from tests.conftest import FakeBackend


def _parallel(fn, n: int) -> list:
    barrier = threading.Barrier(n)

    def run(i):
        barrier.wait()
        try:
            return fn(i)
        except Exception as err:
            return err

    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(run, range(n)))


def test_parallel_calls_never_exceed_the_request_budget(settings):
    s = replace(settings, max_llm_calls_per_run=10)
    llm = LLM(s, backends={"fake": FakeBackend(responder=lambda req: "ok")}, sleep=lambda x: None)
    results = _parallel(lambda i: llm.complete(f"prompt {i}"), 25)
    assert sum(r == "ok" for r in results) == 10
    assert sum(isinstance(r, LLMBudgetExceeded) for r in results) == 15
    assert llm.usage.requests == 10


def test_each_thread_sees_its_own_usage(settings):
    s = replace(settings, max_llm_calls_per_run=50)
    llm = LLM(s, backends={"fake": FakeBackend(responder=lambda req: "ok")}, sleep=lambda x: None)

    def work(i):
        for j in range(i + 1):
            llm.complete(f"thread {i} call {j}")
        return llm.thread_usage()

    usage = _parallel(work, 4)
    assert [u[0] for u in usage] == [1, 2, 3, 4]
    assert all(tokens == 15 * requests for requests, tokens in usage)  # 10 + 5 per fake call


def test_parallel_llm_calls_queue_for_the_throttle(settings):
    s = replace(settings, llm_rpm=60, max_llm_calls_per_run=50)  # one call per second
    sleeps: list[float] = []
    lock = threading.Lock()

    def sleep(seconds):
        with lock:
            sleeps.append(seconds)

    llm = LLM(
        s,
        backends={"fake": FakeBackend(responder=lambda req: "ok")},
        sleep=sleep,
        clock=lambda: 100.0,  # time stands still, so every slot is a wait
    )
    _parallel(lambda i: llm.complete(f"p{i}"), 5)
    assert sorted(sleeps) == [1.0, 2.0, 3.0, 4.0]


def test_parallel_requests_to_one_host_get_separate_turns(settings):
    runtime.configure(settings=settings)
    sleeps: list[float] = []
    lock = threading.Lock()

    def sleep(seconds):
        with lock:
            sleeps.append(seconds)

    client = http.HttpClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok")),
        sleep=sleep,
        clock=lambda: 0.0,
    )
    _parallel(lambda i: client.wait_turn("https://example.com/a"), 4)
    assert sorted(sleeps) == [1.0, 2.0, 3.0]


def test_concurrent_cache_writes_leave_a_whole_file(settings):
    runtime.configure(settings=settings)
    big = "x" * 200_000
    _parallel(lambda i: cache.put("test", {"k": 1}, {"writer": i, "data": big}), 8)
    value = cache.get("test", {"k": 1})
    assert value["data"] == big
    folder = settings.cache_dir / "test"
    assert not [p for p in folder.rglob("*.tmp")]
    assert all(json.loads(p.read_text())["value"] for p in folder.rglob("*.json"))
