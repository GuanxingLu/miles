from argparse import Namespace
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
import requests

from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, default_process_fn
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def make_router_args(
    router_port: int,
    concurrency: int = 10,
    num_gpus: int = 1,
    num_gpus_per_engine: int = 1,
    health_check_interval: float = 1.0,
    health_check_failure_threshold: int = 3,
    max_connections: int | None = None,
    timeout: float | None = None,
) -> Namespace:
    return Namespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=router_port,
        sglang_server_concurrency=concurrency,
        rollout_num_gpus=num_gpus,
        rollout_num_gpus_per_engine=num_gpus_per_engine,
        rollout_health_check_interval=health_check_interval,
        miles_router_health_check_failure_threshold=health_check_failure_threshold,
        miles_router_max_connections=max_connections,
        miles_router_timeout=timeout,
        miles_router_middleware_paths=[],
    )


@contextmanager
def with_miles_router(args: Namespace):
    router = MilesRouter(args, verbose=False)
    server = UvicornThreadServer(router.app, host=args.sglang_router_ip, port=args.sglang_router_port)
    try:
        server.start()
        yield router, server
    finally:
        server.stop()


@contextmanager
def with_mock_worker(host: str = "127.0.0.1", port: int | None = None, latency: float = 0.0):
    port = port or find_available_port(30000)
    server = MockSGLangServer(
        model_name="Qwen/Qwen3-0.6B",
        process_fn=default_process_fn,
        host=host,
        port=port,
        latency=latency,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()


class TestWorkerManagement:
    def test_add_worker_via_query_param(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_miles_router(args) as (router, server):
            worker_url = "http://127.0.0.1:30001"
            r = requests.post(f"{server.url}/add_worker", params={"url": worker_url}, timeout=5.0)
            r.raise_for_status()

            assert r.json()["status"] == "success"
            assert worker_url in router.worker_request_counts
            assert router.worker_request_counts[worker_url] == 0

    def test_add_worker_via_body(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_miles_router(args) as (router, server):
            worker_url = "http://127.0.0.1:30002"
            r = requests.post(f"{server.url}/add_worker", json={"url": worker_url}, timeout=5.0)
            r.raise_for_status()

            assert r.json()["status"] == "success"
            assert worker_url in router.worker_request_counts

    def test_add_worker_duplicate(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_miles_router(args) as (router, server):
            worker_url = "http://127.0.0.1:30003"
            r1 = requests.post(f"{server.url}/add_worker", params={"url": worker_url}, timeout=5.0)
            r1.raise_for_status()

            r2 = requests.post(f"{server.url}/add_worker", params={"url": worker_url}, timeout=5.0)
            r2.raise_for_status()

            assert len(router.worker_request_counts) == 1
            assert worker_url in router.worker_request_counts

    def test_add_worker_missing_url(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_miles_router(args) as (_, server):
            r = requests.post(f"{server.url}/add_worker", json={}, timeout=5.0)
            assert r.status_code == 400
            assert "error" in r.json()

    def test_list_workers(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_miles_router(args) as (_, server):
            worker_urls = ["http://127.0.0.1:30001", "http://127.0.0.1:30002"]
            for url in worker_urls:
                requests.post(f"{server.url}/add_worker", params={"url": url}, timeout=5.0)

            r = requests.get(f"{server.url}/list_workers", timeout=5.0)
            r.raise_for_status()

            listed = r.json()["urls"]
            assert set(listed) == set(worker_urls)


class TestLoadBalancing:
    def test_use_url_selects_min_load(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)
        router = MilesRouter(args, verbose=False)

        router.worker_request_counts = {
            "http://w1:8000": 5,
            "http://w2:8000": 2,
            "http://w3:8000": 8,
        }

        selected = router._use_url()
        assert selected == "http://w2:8000"
        assert router.worker_request_counts["http://w2:8000"] == 3

    def test_use_url_excludes_dead_workers(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)
        router = MilesRouter(args, verbose=False)

        router.worker_request_counts = {
            "http://w1:8000": 5,
            "http://w2:8000": 1,
            "http://w3:8000": 3,
        }
        router.dead_workers = {"http://w2:8000"}

        selected = router._use_url()
        assert selected == "http://w3:8000"
        assert router.worker_request_counts["http://w3:8000"] == 4

    def test_use_url_raises_when_all_dead(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)
        router = MilesRouter(args, verbose=False)

        router.worker_request_counts = {"http://w1:8000": 0}
        router.dead_workers = {"http://w1:8000"}

        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._use_url()

    def test_finish_url_decrements_count(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)
        router = MilesRouter(args, verbose=False)

        router.worker_request_counts = {"http://w1:8000": 5}

        router._finish_url("http://w1:8000")
        assert router.worker_request_counts["http://w1:8000"] == 4

    def test_finish_url_raises_on_unknown(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)
        router = MilesRouter(args, verbose=False)

        with pytest.raises(AssertionError, match="not recognized"):
            router._finish_url("http://unknown:8000")


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_check_worker_health_success(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_mock_worker() as worker:
            router = MilesRouter(args, verbose=False)
            url, healthy = await router._check_worker_health(worker.url)

            assert url == worker.url
            assert healthy is True

    @pytest.mark.asyncio
    async def test_check_worker_health_failure(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)
        router = MilesRouter(args, verbose=False)

        url, healthy = await router._check_worker_health("http://127.0.0.1:59999")

        assert url == "http://127.0.0.1:59999"
        assert healthy is False

    @pytest.mark.asyncio
    async def test_health_check_marks_dead_worker(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port, health_check_failure_threshold=2)
        router = MilesRouter(args, verbose=False)

        bad_url = "http://127.0.0.1:59998"
        router.worker_request_counts[bad_url] = 0
        router.worker_failure_counts[bad_url] = 0

        with patch.object(router, "_check_worker_health", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = (bad_url, False)

            await router._check_worker_health(bad_url)
            router.worker_failure_counts[bad_url] += 1
            assert bad_url not in router.dead_workers

            await router._check_worker_health(bad_url)
            router.worker_failure_counts[bad_url] += 1
            if router.worker_failure_counts[bad_url] >= args.miles_router_health_check_failure_threshold:
                router.dead_workers.add(bad_url)

            assert bad_url in router.dead_workers

    @pytest.mark.asyncio
    async def test_health_check_resets_on_success(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port, health_check_failure_threshold=3)
        router = MilesRouter(args, verbose=False)

        url = "http://127.0.0.1:59997"
        router.worker_request_counts[url] = 0
        router.worker_failure_counts[url] = 2

        with patch.object(router, "_check_worker_health", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = (url, True)

            _, is_healthy = await router._check_worker_health(url)
            if is_healthy:
                router.worker_failure_counts[url] = 0

            assert router.worker_failure_counts[url] == 0


class TestProxyIntegration:
    def test_proxy_forwards_request(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_mock_worker() as worker:
            with with_miles_router(args) as (_, router_server):
                r = requests.post(f"{router_server.url}/add_worker", params={"url": worker.url}, timeout=5.0)
                r.raise_for_status()

                r = requests.post(
                    f"{router_server.url}/generate",
                    json={"input_ids": [1, 2, 3], "return_logprob": True},
                    timeout=10.0,
                )
                r.raise_for_status()

                assert "text" in r.json()
                assert len(worker.request_log) == 1

    def test_proxy_load_balances(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_mock_worker() as worker1:
            with with_mock_worker() as worker2:
                with with_miles_router(args) as (_, router_server):
                    requests.post(f"{router_server.url}/add_worker", params={"url": worker1.url}, timeout=5.0)
                    requests.post(f"{router_server.url}/add_worker", params={"url": worker2.url}, timeout=5.0)

                    for _ in range(4):
                        r = requests.post(
                            f"{router_server.url}/generate",
                            json={"input_ids": [1, 2, 3], "return_logprob": True},
                            timeout=10.0,
                        )
                        r.raise_for_status()

                    assert len(worker1.request_log) == 2
                    assert len(worker2.request_log) == 2

    def test_proxy_health_endpoint(self):
        router_port = find_available_port(20000)
        args = make_router_args(router_port)

        with with_mock_worker() as worker:
            with with_miles_router(args) as (_, router_server):
                requests.post(f"{router_server.url}/add_worker", params={"url": worker.url}, timeout=5.0)

                r = requests.get(f"{router_server.url}/health", timeout=5.0)
                r.raise_for_status()
                assert r.json()["status"] == "ok"
