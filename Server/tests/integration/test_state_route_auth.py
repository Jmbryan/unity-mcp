"""Auth guard for the /agent-status and /lease/release custom routes.

Remote-hosted mode must require the same credential as the rest of the HTTP
surface (a valid X-API-Key validated through ApiKeyService, fail closed).
Local mode has no credential mechanism: the routes stay open while the server
binds loopback; a non-loopback bind accepts only loopback peers.
"""

import types

import pytest

from core.config import config
from services.api_key_service import ApiKeyService, ValidationResult


def _authorize_state_route(request):
    # Imported lazily: a module-level `import main` reconfigures logging for
    # the whole pytest process and breaks the logging characterization tests.
    from main import _authorize_state_route as impl

    return impl(request)


def _is_loopback_host(value):
    from main import _is_loopback_host as impl

    return impl(value)


class _FakeRequest:
    def __init__(self, headers=None, client_host=None):
        self.headers = headers or {}
        self.client = (
            types.SimpleNamespace(host=client_host, port=12345)
            if client_host is not None
            else None
        )


class _FakeApiKeyService:
    def __init__(self, result):
        self._result = result
        self.validated_keys = []

    async def validate(self, api_key):
        self.validated_keys.append(api_key)
        return self._result


@pytest.fixture(autouse=True)
def _restore_auth_state(monkeypatch):
    monkeypatch.setattr(config, "http_remote_hosted", config.http_remote_hosted)
    previous = ApiKeyService._instance
    yield
    ApiKeyService._instance = previous


def _install_service(result):
    service = _FakeApiKeyService(result)
    ApiKeyService._instance = service
    return service


class TestLoopbackHelper:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "[::1]", "LOCALHOST"])
    def test_loopback_hosts(self, host):
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", "", None])
    def test_non_loopback_hosts(self, host):
        assert _is_loopback_host(host) is False


class TestRemoteHostedAuth:
    @pytest.mark.asyncio
    async def test_missing_key_is_401(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", True)
        _install_service(ValidationResult(valid=True, user_id="user-1"))

        denied = await _authorize_state_route(_FakeRequest())

        assert denied is not None
        assert denied.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_is_403(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", True)
        _install_service(ValidationResult(valid=False, error="nope"))

        denied = await _authorize_state_route(
            _FakeRequest(headers={"X-API-Key": "bad-key"}))

        assert denied is not None
        assert denied.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_key_without_user_id_is_403(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", True)
        _install_service(ValidationResult(valid=True, user_id=None))

        denied = await _authorize_state_route(
            _FakeRequest(headers={"X-API-Key": "key"}))

        assert denied is not None
        assert denied.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", True)
        service = _install_service(ValidationResult(valid=True, user_id="user-1"))

        denied = await _authorize_state_route(
            _FakeRequest(headers={"X-API-Key": "good-key"}))

        assert denied is None
        assert service.validated_keys == ["good-key"]

    @pytest.mark.asyncio
    async def test_uninitialized_auth_service_fails_closed(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", True)
        ApiKeyService._instance = None

        denied = await _authorize_state_route(
            _FakeRequest(headers={"X-API-Key": "key"}))

        assert denied is not None
        assert denied.status_code == 503

    @pytest.mark.asyncio
    async def test_validation_error_fails_closed(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", True)

        class _Exploding:
            async def validate(self, api_key):
                raise RuntimeError("auth backend down")

        ApiKeyService._instance = _Exploding()

        denied = await _authorize_state_route(
            _FakeRequest(headers={"X-API-Key": "key"}))

        assert denied is not None
        assert denied.status_code == 503


class TestLocalMode:
    @pytest.mark.asyncio
    async def test_loopback_bind_stays_open(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", False)
        monkeypatch.setenv("UNITY_MCP_HTTP_HOST", "localhost")

        denied = await _authorize_state_route(_FakeRequest())

        assert denied is None

    @pytest.mark.asyncio
    async def test_default_bind_is_loopback_and_open(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", False)
        monkeypatch.delenv("UNITY_MCP_HTTP_HOST", raising=False)

        denied = await _authorize_state_route(_FakeRequest())

        assert denied is None

    @pytest.mark.asyncio
    async def test_non_loopback_bind_rejects_remote_peer(self, monkeypatch):
        monkeypatch.setattr(config, "http_remote_hosted", False)
        monkeypatch.setenv("UNITY_MCP_HTTP_HOST", "0.0.0.0")

        denied = await _authorize_state_route(
            _FakeRequest(client_host="192.168.1.99"))

        assert denied is not None
        assert denied.status_code == 403

    @pytest.mark.asyncio
    async def test_non_loopback_bind_allows_loopback_peer(self, monkeypatch):
        """The bridge on the same machine keeps its force-release lever."""
        monkeypatch.setattr(config, "http_remote_hosted", False)
        monkeypatch.setenv("UNITY_MCP_HTTP_HOST", "0.0.0.0")

        denied = await _authorize_state_route(
            _FakeRequest(client_host="127.0.0.1"))

        assert denied is None
