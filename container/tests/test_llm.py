"""Tests for app.llm -- client and providers."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock litellm before importing any app.llm modules
_litellm_mock = MagicMock()


class _AuthenticationError(Exception):
    pass


_litellm_mock.exceptions.AuthenticationError = _AuthenticationError
sys.modules.setdefault("litellm", _litellm_mock)

from app.llm.providers import (  # noqa: E402
    extract_provider,
    get_api_key,
    get_base_url,
    resolve_provider_params,
    PROVIDER_SECRET_MAP,
    LOCAL_PROVIDERS,
)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

class TestExtractProvider:

    def test_extract_provider_from_slashed_model(self):
        assert extract_provider("openrouter/openai/gpt-4o-mini") == "openrouter"

    def test_extract_provider_from_groq(self):
        assert extract_provider("groq/llama3-70b") == "groq"

    def test_extract_provider_defaults_to_openai(self):
        assert extract_provider("gpt-4") == "openai"

    def test_extract_provider_ollama(self):
        assert extract_provider("ollama/llama3") == "ollama"


class TestGetApiKey:

    @patch("app.llm.providers.retrieve_secret", new_callable=AsyncMock, return_value="sk-test-key")
    async def test_get_api_key_returns_key(self, mock_retrieve):
        key = await get_api_key("openrouter")
        assert key == "sk-test-key"
        mock_retrieve.assert_awaited_once_with("openrouter_api_key")

    @patch("app.llm.providers.retrieve_secret", new_callable=AsyncMock, return_value=None)
    async def test_get_api_key_returns_none_for_missing_key(self, mock_retrieve):
        key = await get_api_key("openrouter")
        assert key is None

    async def test_get_api_key_returns_none_for_local_provider(self):
        key = await get_api_key("ollama")
        assert key is None

    async def test_get_api_key_returns_none_for_unknown_provider(self):
        key = await get_api_key("unknown-provider")
        assert key is None


class TestGetBaseUrl:

    @patch("app.llm.providers.SettingsRepository")
    async def test_get_base_url_ollama(self, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://ollama:11434")
        url = await get_base_url("ollama")
        assert url == "http://ollama:11434"

    async def test_get_base_url_returns_none_for_non_ollama(self):
        url = await get_base_url("openrouter")
        assert url is None


class TestResolveProviderParams:

    @patch("app.llm.providers.retrieve_secret", new_callable=AsyncMock, return_value="sk-key")
    async def test_resolve_openrouter_includes_api_key(self, mock_retrieve):
        params = await resolve_provider_params("openrouter/openai/gpt-4o")
        assert params["api_key"] == "sk-key"

    @patch("app.llm.providers.SettingsRepository")
    async def test_resolve_ollama_includes_base_url(self, mock_settings):
        mock_settings.get_value = AsyncMock(return_value="http://localhost:11434")
        params = await resolve_provider_params("ollama/llama3")
        assert params["api_base"] == "http://localhost:11434"
        assert "api_key" not in params


# ---------------------------------------------------------------------------
# LLM complete function
# ---------------------------------------------------------------------------

class TestLLMComplete:

    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_calls_litellm(self, mock_repo, mock_params, mock_acompletion):
        mock_repo.get = AsyncMock(return_value={
            "agent_id": "light-agent",
            "enabled": True,
            "model": "openrouter/openai/gpt-4o-mini",
            "timeout": 5,
            "max_iterations": 3,
            "temperature": 0.7,
            "max_tokens": 256,
            "description": "Light agent",
        })
        choice = MagicMock()
        choice.message.content = "Done!"
        mock_acompletion.return_value = MagicMock(choices=[choice])

        from app.llm.client import complete
        result = await complete("light-agent", [{"role": "user", "content": "turn on light"}])
        assert result == "Done!"
        mock_acompletion.assert_awaited_once()

    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_raises_on_missing_config(self, mock_repo, mock_params):
        mock_repo.get = AsyncMock(return_value=None)
        from app.llm.client import complete
        with pytest.raises(ValueError, match="No config found"):
            await complete("nonexistent-agent", [{"role": "user", "content": "hi"}])

    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_raises_on_no_model(self, mock_repo, mock_params):
        mock_repo.get = AsyncMock(return_value={
            "agent_id": "test-agent",
            "enabled": True,
            "model": None,
            "timeout": 5,
            "max_iterations": 3,
            "temperature": 0.7,
            "max_tokens": 256,
            "description": "No model",
        })
        from app.llm.client import complete
        with pytest.raises(ValueError, match="No model configured"):
            await complete("test-agent", [{"role": "user", "content": "hi"}])

    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_passes_overrides(self, mock_repo, mock_params, mock_acompletion):
        mock_repo.get = AsyncMock(return_value={
            "agent_id": "test-agent",
            "enabled": True,
            "model": "openrouter/openai/gpt-4o",
            "timeout": 5,
            "max_iterations": 3,
            "temperature": 0.5,
            "max_tokens": 100,
            "description": "test",
        })
        choice = MagicMock()
        choice.message.content = "result"
        mock_acompletion.return_value = MagicMock(choices=[choice])

        from app.llm.client import complete
        await complete("test-agent", [{"role": "user", "content": "test"}], temperature=0.1)
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.1 or call_kwargs[1].get("temperature") == 0.1

    @patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API Error"))
    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_propagates_llm_error(self, mock_repo, mock_params, mock_acompletion):
        mock_repo.get = AsyncMock(return_value={
            "agent_id": "test-agent",
            "enabled": True,
            "model": "openrouter/openai/gpt-4o",
            "timeout": 5,
            "max_iterations": 3,
            "temperature": 0.7,
            "max_tokens": 256,
            "description": "test",
        })
        from app.llm.client import complete
        with pytest.raises(Exception, match="API Error"):
            await complete("test-agent", [{"role": "user", "content": "test"}])

    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_retries_once_on_empty_response(self, mock_repo, mock_params, mock_acompletion):
        mock_repo.get = AsyncMock(return_value={
            "agent_id": "light-agent",
            "enabled": True,
            "model": "openrouter/openai/gpt-4o-mini",
            "timeout": 5,
            "max_iterations": 3,
            "temperature": 0.2,
            "max_tokens": 256,
            "description": "Light agent",
        })
        empty_choice = MagicMock()
        empty_choice.message.content = ""
        empty_choice.finish_reason = "length"
        empty_response = MagicMock(choices=[empty_choice])

        valid_choice = MagicMock()
        valid_choice.message.content = "Light is on!"
        valid_response = MagicMock(choices=[valid_choice])

        mock_acompletion.side_effect = [empty_response, valid_response]

        from app.llm.client import complete
        result = await complete("light-agent", [{"role": "user", "content": "turn on light"}])
        assert result == "Light is on!"
        assert mock_acompletion.await_count == 2

    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.llm.client.resolve_provider_params", new_callable=AsyncMock, return_value={})
    @patch("app.llm.client.AgentConfigRepository")
    async def test_complete_returns_empty_after_retry_exhausted(self, mock_repo, mock_params, mock_acompletion):
        mock_repo.get = AsyncMock(return_value={
            "agent_id": "light-agent",
            "enabled": True,
            "model": "openrouter/openai/gpt-4o-mini",
            "timeout": 5,
            "max_iterations": 3,
            "temperature": 0.2,
            "max_tokens": 256,
            "description": "Light agent",
        })
        empty_choice = MagicMock()
        empty_choice.message.content = ""
        empty_choice.finish_reason = "length"
        empty_response = MagicMock(choices=[empty_choice])

        mock_acompletion.side_effect = [empty_response, empty_response]

        from app.llm.client import complete
        result = await complete("light-agent", [{"role": "user", "content": "turn on light"}])
        assert result == ""
        assert mock_acompletion.await_count == 2
