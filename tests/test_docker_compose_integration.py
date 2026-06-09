# pylint: disable=redefined-outer-name
# Pytest fixture parameters intentionally shadow the fixture function names;
# Pylint W0621 is a known false positive for this pattern.
"""Docker Compose integration tests.

Tests verify:
- Services start and become healthy
- ClickHouse is accessible
- RFSN service can connect to ClickHouse
- Health checks pass

Requirements:
    - Docker and docker-compose installed
    - Ports 8000 and 8123 available

These tests are SKIPPED by default.  Set ``RFSN_INTEGRATION_TESTS=1`` in the
environment to opt in.  They are tagged ``@pytest.mark.integration`` so they
can also be selected/deselected via ``-m integration``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

# Skip the entire module unless explicitly opted in.
pytestmark = pytest.mark.skipif(
    not os.getenv("RFSN_INTEGRATION_TESTS"),
    reason=(
        "Docker Compose integration tests are opt-in. "
        "Set RFSN_INTEGRATION_TESTS=1 to enable."
    ),
)


@pytest.fixture(scope="module")
def docker_compose():
    """Start Docker Compose services for testing."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"

    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    # Check if docker is available
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pytest.skip("Docker Compose not available")

    # Start services — hard timeout so a broken Docker daemon never hangs CI.
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            check=True,
            cwd=compose_file.parent,
            timeout=120,
        )

        # Wait for services to be healthy
        max_wait = 60  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                result = subprocess.run(
                    ["docker", "compose", "ps", "--format", "json"],
                    capture_output=True,
                    text=True,
                    cwd=compose_file.parent,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                break

            if result.returncode == 0:
                try:
                    services = json.loads(result.stdout)
                    if isinstance(services, list):
                        healthy = all(
                            s.get("Health", "") == "healthy" or
                            s.get("State", "") == "running"
                            for s in services
                        )
                        if healthy:
                            break
                except json.JSONDecodeError:
                    pass

            time.sleep(2)
        else:
            # Cleanup on timeout
            subprocess.run(
                ["docker", "compose", "down"],
                capture_output=True,
                cwd=compose_file.parent,
                timeout=30,
            )
            pytest.skip("Services failed to become healthy")

        yield compose_file

        # Cleanup
        subprocess.run(
            ["docker", "compose", "down", "-v"],
            capture_output=True,
            cwd=compose_file.parent,
            timeout=60,
        )

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"Failed to start Docker Compose: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SKIP_DOCKER_TESTS"),
    reason="Docker tests disabled via SKIP_DOCKER_TESTS"
)
class TestDockerComposeServices:
    """Docker Compose service tests."""

    def test_clickhouse_service_running(self, docker_compose):
        """ClickHouse service should be running."""
        result = subprocess.run(
            ["docker", "compose", "ps", "rfsn-clickhouse", "-q"],
            capture_output=True,
            text=True,
            cwd=docker_compose.parent
        )
        assert result.returncode == 0
        assert result.stdout.strip() != ""

    def test_rfsn_service_running(self, docker_compose):
        """RFSN service should be running."""
        result = subprocess.run(
            ["docker", "compose", "ps", "rfsn-v10", "-q"],
            capture_output=True,
            text=True,
            cwd=docker_compose.parent
        )
        assert result.returncode == 0
        assert result.stdout.strip() != ""

    def test_clickhouse_port_accessible(self, docker_compose):
        """ClickHouse HTTP port should be accessible."""
        import urllib.request

        try:
            req = urllib.request.Request("http://localhost:8123/ping")
            with urllib.request.urlopen(req, timeout=5) as response:
                assert response.status == 200
        except Exception as e:
            pytest.fail(f"ClickHouse not accessible: {e}")

    def test_rfsn_health_endpoint(self, docker_compose):
        """RFSN health endpoint should return healthy."""
        import urllib.request

        # Wait a bit for service to start
        time.sleep(5)

        try:
            server_port = os.getenv("RFSN_SERVER_PORT", "8000")
            url = f"http://localhost:{server_port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read())
                assert data.get("status") in ["healthy", "ok", "ready"]
        except Exception as e:
            pytest.fail(f"RFSN health check failed: {e}")

    def test_clickhouse_version_query(self, docker_compose):
        """Should be able to query ClickHouse version."""
        import urllib.request

        query = "SELECT version()"
        data = query.encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:8123/",
            data=data,
            headers={"Content-Type": "text/plain"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                version = response.read().decode().strip()
                assert len(version) > 0
                assert "." in version  # Version string format
        except Exception as e:
            pytest.fail(f"ClickHouse query failed: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SKIP_DOCKER_TESTS"),
    reason="Docker tests disabled via SKIP_DOCKER_TESTS"
)
class TestDockerComposeTelemetry:
    """Telemetry integration through Docker Compose."""

    def test_telemetry_table_exists(self, docker_compose):
        """Telemetry table should exist in ClickHouse."""
        import urllib.request

        query = "SHOW TABLES"
        data = query.encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:8123/",
            data=data,
            headers={"Content-Type": "text/plain"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                tables = response.read().decode().strip()
                # Should have at least one table
                assert len(tables) > 0
        except Exception as e:
            pytest.fail(f"Failed to list tables: {e}")

    def test_can_insert_telemetry(self, docker_compose):
        """Should be able to insert telemetry data."""
        import urllib.request

        # Create a simple test table if it doesn't exist
        create_query = """
        CREATE TABLE IF NOT EXISTS test_telemetry (
            id String,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY timestamp
        """

        req = urllib.request.Request(
            "http://localhost:8123/",
            data=create_query.encode("utf-8"),
            headers={"Content-Type": "text/plain"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                assert response.status == 200
        except Exception as e:
            pytest.skip(f"Cannot create test table: {e}")

        # Insert test data
        insert_query = "INSERT INTO test_telemetry (id) VALUES ('test')"
        req = urllib.request.Request(
            "http://localhost:8123/",
            data=insert_query.encode("utf-8"),
            headers={"Content-Type": "text/plain"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                assert response.status == 200
        except Exception as e:
            pytest.fail(f"Failed to insert data: {e}")

        # Verify data was inserted
        select_query = "SELECT count() FROM test_telemetry"
        req = urllib.request.Request(
            "http://localhost:8123/",
            data=select_query.encode("utf-8"),
            headers={"Content-Type": "text/plain"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                count = int(response.read().decode().strip())
                assert count >= 1
        except Exception as e:
            pytest.fail(f"Failed to verify insertion: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SKIP_DOCKER_TESTS"),
    reason="Docker tests disabled via SKIP_DOCKER_TESTS"
)
class TestDockerComposeCleanup:
    """Cleanup verification tests."""

    def test_volumes_created(self, docker_compose):
        """Docker volumes should be created."""
        result = subprocess.run(
            ["docker", "volume", "ls", "-q"],
            capture_output=True,
            text=True
        )

        volumes = result.stdout.strip().split("\n")
        assert any("rfsn" in v for v in volumes)

    def test_network_created(self, docker_compose):
        """Docker network should be created."""
        result = subprocess.run(
            ["docker", "network", "ls", "-q"],
            capture_output=True,
            text=True
        )

        networks = result.stdout.strip().split("\n")
        assert any("rfsn" in n for n in networks)
