"""Pinned per-thread model banner for Telegram topics.

/model is already session-scoped (session keys include thread_id). These tests
cover the Telegram-only side effect: create/update a pinned banner so the
active model is visible in the topic without scrolling history.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _install_fake_telegram(monkeypatch):
    fake_telegram = types.ModuleType("telegram")
    fake_telegram.Update = SimpleNamespace(ALL_TYPES=())
    fake_telegram.Bot = object
    fake_telegram.Message = object
    fake_telegram.InlineKeyboardButton = object
    fake_telegram.InlineKeyboardMarkup = object

    fake_error = types.ModuleType("telegram.error")
    fake_error.NetworkError = type("NetworkError", (Exception,), {})
    fake_error.BadRequest = type("BadRequest", (Exception,), {})
    fake_error.TimedOut = type("TimedOut", (Exception,), {})
    fake_telegram.error = fake_error

    fake_constants = types.ModuleType("telegram.constants")
    fake_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    fake_constants.ChatType = SimpleNamespace(
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
        PRIVATE="private",
    )
    fake_telegram.constants = fake_constants

    fake_ext = types.ModuleType("telegram.ext")
    fake_ext.Application = object
    fake_ext.CommandHandler = object
    fake_ext.CallbackQueryHandler = object
    fake_ext.MessageHandler = object
    fake_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    fake_ext.filters = object

    fake_request = types.ModuleType("telegram.request")
    fake_request.HTTPXRequest = object

    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", fake_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", fake_request)


@pytest.fixture
def adapter(monkeypatch):
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    a._bot.pin_chat_message = AsyncMock()
    a.send = AsyncMock()
    a.edit_message = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_upsert_sends_and_pins_on_first_call(adapter):
    adapter.send.return_value = SendResult(success=True, message_id="42")

    message_id = await adapter.upsert_thread_model_pin(
        "-1001",
        model="gpt-5.6",
        provider="OpenAI",
        thread_id="5363",
    )

    assert message_id == "42"
    adapter.send.assert_awaited_once()
    send_kwargs = adapter.send.await_args
    assert send_kwargs.args[0] == "-1001"
    assert "gpt-5.6" in send_kwargs.args[1]
    assert "OpenAI" in send_kwargs.args[1]
    assert send_kwargs.kwargs["metadata"] == {"thread_id": "5363"}
    adapter._bot.pin_chat_message.assert_awaited_once()
    pin_kwargs = adapter._bot.pin_chat_message.await_args.kwargs
    assert pin_kwargs["message_id"] == 42
    assert pin_kwargs["disable_notification"] is True
    assert adapter._thread_model_pin_ids[("-1001", "5363")] == "42"


@pytest.mark.asyncio
async def test_upsert_edits_cached_pin_in_place(adapter):
    adapter._thread_model_pin_ids[("-1001", "5363")] = "42"
    adapter.edit_message.return_value = SendResult(success=True, message_id="42")

    message_id = await adapter.upsert_thread_model_pin(
        "-1001",
        model="claude-opus-4-6",
        provider="Anthropic",
        thread_id="5363",
    )

    assert message_id == "42"
    adapter.edit_message.assert_awaited_once()
    edit_args = adapter.edit_message.await_args
    assert edit_args.args[0] == "-1001"
    assert edit_args.args[1] == "42"
    assert "claude-opus-4-6" in edit_args.args[2]
    adapter.send.assert_not_awaited()
    adapter._bot.pin_chat_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_resends_when_edit_fails(adapter):
    adapter._thread_model_pin_ids[("-1001", "5363")] = "42"
    adapter.edit_message.return_value = SendResult(success=False)
    adapter.send.return_value = SendResult(success=True, message_id="99")

    message_id = await adapter.upsert_thread_model_pin(
        "-1001",
        model="gpt-5.6",
        thread_id="5363",
    )

    assert message_id == "99"
    adapter.send.assert_awaited_once()
    assert adapter._thread_model_pin_ids[("-1001", "5363")] == "99"


@pytest.mark.asyncio
async def test_upsert_keeps_message_when_pin_fails(adapter):
    adapter.send.return_value = SendResult(success=True, message_id="7")
    adapter._bot.pin_chat_message.side_effect = RuntimeError("not admin")

    message_id = await adapter.upsert_thread_model_pin(
        "12345",
        model="gpt-5.6",
    )

    assert message_id == "7"
    assert adapter._thread_model_pin_ids[("12345", "")] == "7"


@pytest.mark.asyncio
async def test_upsert_scopes_pins_per_thread(adapter):
    adapter.send.side_effect = [
        SendResult(success=True, message_id="1"),
        SendResult(success=True, message_id="2"),
    ]

    await adapter.upsert_thread_model_pin("-1001", model="a", thread_id="10")
    await adapter.upsert_thread_model_pin("-1001", model="b", thread_id="20")

    assert adapter._thread_model_pin_ids[("-1001", "10")] == "1"
    assert adapter._thread_model_pin_ids[("-1001", "20")] == "2"


@pytest.mark.asyncio
async def test_helper_skips_non_telegram_and_missing_adapter():
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _s: None

    discord_source = SessionSource(
        platform=Platform.DISCORD, chat_id="1", chat_type="dm"
    )
    await runner._update_telegram_thread_model_pin(
        discord_source, model="gpt-5.6", provider="x"
    )

    tg_source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", thread_id="9"
    )
    await runner._update_telegram_thread_model_pin(
        tg_source, model="gpt-5.6", provider="x"
    )


@pytest.mark.asyncio
async def test_helper_forwards_thread_id_to_adapter():
    pin_fn = AsyncMock()
    adapter = SimpleNamespace(upsert_thread_model_pin=pin_fn)
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _s: adapter

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003763717268",
        chat_type="group",
        thread_id="5363",
    )
    await runner._update_telegram_thread_model_pin(
        source, model="gpt-5.6", provider="custom:cloudflare-aig-openai"
    )

    pin_fn.assert_awaited_once_with(
        "-1003763717268",
        model="gpt-5.6",
        provider="custom:cloudflare-aig-openai",
        thread_id="5363",
    )


def _fake_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.6",
        target_provider="custom:cloudflare-aig-openai",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://example.invalid/openai",
        api_mode="codex_responses",
        provider_label="Cloudflare AIG OpenAI",
        is_global=False,
    )


@pytest.mark.asyncio
async def test_model_command_updates_telegram_thread_pin(tmp_path, monkeypatch):
    """Typed /model --session must upsert the topic pin on Telegram sources."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "old-model",
                    "provider": "openrouter",
                },
                "providers": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    pin_fn = AsyncMock(return_value="pin-1")
    adapter = SimpleNamespace(upsert_thread_model_pin=pin_fn)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._adapter_for_source = lambda _s: adapter

    event = MessageEvent(
        text="/model gpt-5.6 --session --provider custom:cloudflare-aig-openai",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1003763717268",
            chat_type="group",
            thread_id="5363",
        ),
    )

    result = await runner._handle_model_command(event)

    assert result is not None
    assert "gpt-5.6" in result
    pin_fn.assert_awaited_once()
    kwargs = pin_fn.await_args.kwargs
    assert kwargs["model"] == "gpt-5.6"
    assert "cloudflare" in kwargs["provider"].lower() or kwargs["provider"]
    assert kwargs["thread_id"] == "5363"
    assert pin_fn.await_args.args[0] == "-1003763717268"
