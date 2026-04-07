"""Tests for /thread gateway slash command.

Tests the _handle_thread_command handler which creates, lists, and closes
REAL Telegram forum topics via the Bot API.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, ForumThreadStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(text="/thread", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890", chat_type="group"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        chat_type=chat_type,
    )
    return MessageEvent(text=text, source=source)


def _make_mock_bot():
    """Create a mock Telegram Bot with forum topic methods."""
    bot = AsyncMock()

    # create_forum_topic returns an object with message_thread_id
    topic_result = MagicMock()
    topic_result.message_thread_id = 42
    bot.create_forum_topic = AsyncMock(return_value=topic_result)
    bot.close_forum_topic = AsyncMock()

    return bot


def _make_runner(tmp_path, bot=None):
    """Create a bare GatewayRunner with a ForumThreadStore backed by tmp_path."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._voice_mode = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()

    config = MagicMock(spec=GatewayConfig)
    config.sessions_dir = tmp_path / "sessions"
    config.group_sessions_per_user = True
    runner.config = config

    # Set up adapters with a mock Telegram adapter
    mock_adapter = MagicMock()
    mock_adapter._bot = bot or _make_mock_bot()
    runner.adapters = {Platform.TELEGRAM: mock_adapter}

    runner._thread_store = ForumThreadStore(config.sessions_dir / "forum_threads.json")

    return runner


# ---------------------------------------------------------------------------
# ForumThreadStore unit tests (persistence layer)
# ---------------------------------------------------------------------------


class TestForumThreadStore:
    """Tests for the ForumThreadStore persistence layer in gateway/session.py."""

    def test_import(self):
        from gateway.session import ForumThreadStore
        assert ForumThreadStore is not None

    def test_add_and_get(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        store.add("123", "research", 42)
        assert store.get_topic_id("123", "research") == 42

    def test_get_nonexistent(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        assert store.get_topic_id("123", "nope") is None

    def test_list_threads_empty(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        assert store.list_threads("123") == {}

    def test_list_threads(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        store.add("123", "alpha", 10)
        store.add("123", "beta", 20)
        threads = store.list_threads("123")
        assert threads == {"alpha": 10, "beta": 20}

    def test_remove_thread(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        store.add("123", "alpha", 10)
        store.add("123", "beta", 20)
        removed_id = store.remove("123", "alpha")
        assert removed_id == 10
        assert "alpha" not in store.list_threads("123")
        assert "beta" in store.list_threads("123")

    def test_remove_nonexistent(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        assert store.remove("123", "nope") is None

    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "threads.json"
        store1 = ForumThreadStore(path)
        store1.add("123", "alpha", 10)
        store1.add("123", "beta", 20)

        store2 = ForumThreadStore(path)
        assert store2.list_threads("123") == {"alpha": 10, "beta": 20}
        assert store2.get_topic_id("123", "alpha") == 10

    def test_different_chats_isolated(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        store.add("aaa", "alpha", 10)
        store.add("bbb", "beta", 20)

        assert store.list_threads("aaa") == {"alpha": 10}
        assert store.list_threads("bbb") == {"beta": 20}

    def test_remove_last_thread_cleans_chat(self, tmp_path):
        store = ForumThreadStore(tmp_path / "threads.json")
        store.add("123", "only", 10)
        store.remove("123", "only")
        assert store.list_threads("123") == {}


# ---------------------------------------------------------------------------
# /thread command handler tests
# ---------------------------------------------------------------------------


class TestHandleThreadCommand:
    """Tests for GatewayRunner._handle_thread_command."""

    @pytest.mark.asyncio
    async def test_no_args_shows_list_empty(self, tmp_path):
        """/thread with no args lists threads (empty case)."""
        runner = _make_runner(tmp_path)
        event = _make_event(text="/thread")
        result = await runner._handle_thread_command(event)
        assert "no managed forum topics" in result.lower()

    @pytest.mark.asyncio
    async def test_list_shows_threads(self, tmp_path):
        """/thread list shows managed forum topics."""
        runner = _make_runner(tmp_path)
        # Pre-populate a thread
        runner._thread_store.add("67890", "research", 42)

        event = _make_event(text="/thread list")
        result = await runner._handle_thread_command(event)
        assert "research" in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_create_thread(self, tmp_path):
        """/thread <name> creates a real Telegram forum topic."""
        bot = _make_mock_bot()
        runner = _make_runner(tmp_path, bot=bot)

        event = _make_event(text="/thread research")
        result = await runner._handle_thread_command(event)

        bot.create_forum_topic.assert_awaited_once_with(chat_id=67890, name="research")
        assert "research" in result
        assert "created" in result.lower()
        # Verify it was stored
        assert runner._thread_store.get_topic_id("67890", "research") == 42

    @pytest.mark.asyncio
    async def test_create_duplicate_thread(self, tmp_path):
        """/thread <name> for existing thread warns instead of creating."""
        runner = _make_runner(tmp_path)
        runner._thread_store.add("67890", "research", 42)

        event = _make_event(text="/thread research")
        result = await runner._handle_thread_command(event)
        assert "already exists" in result.lower()

    @pytest.mark.asyncio
    async def test_close_thread(self, tmp_path):
        """/thread close <name> calls close_forum_topic and removes from store."""
        bot = _make_mock_bot()
        runner = _make_runner(tmp_path, bot=bot)
        runner._thread_store.add("67890", "research", 42)

        event = _make_event(text="/thread close research")
        result = await runner._handle_thread_command(event)

        bot.close_forum_topic.assert_awaited_once_with(chat_id=67890, message_thread_id=42)
        assert "closed" in result.lower()
        assert runner._thread_store.get_topic_id("67890", "research") is None

    @pytest.mark.asyncio
    async def test_close_nonexistent_thread(self, tmp_path):
        """/thread close <nonexistent> returns error."""
        runner = _make_runner(tmp_path)
        result = await runner._handle_thread_command(
            _make_event(text="/thread close nonexistent")
        )
        assert "not found" in result.lower() or "no managed" in result.lower()

    @pytest.mark.asyncio
    async def test_close_no_name(self, tmp_path):
        """/thread close with no name shows usage."""
        runner = _make_runner(tmp_path)
        result = await runner._handle_thread_command(
            _make_event(text="/thread close")
        )
        assert "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_dm_create_succeeds(self, tmp_path):
        """/thread in a DM creates a forum topic (Bot API 9.4+)."""
        bot = _make_mock_bot()
        runner = _make_runner(tmp_path, bot=bot)
        event = _make_event(text="/thread research", chat_type="dm")
        result = await runner._handle_thread_command(event)
        assert "created" in result.lower()
        bot.create_forum_topic.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_telegram_rejected(self, tmp_path):
        """/thread on non-Telegram platform returns error."""
        runner = _make_runner(tmp_path)
        event = _make_event(text="/thread research", platform=Platform.DISCORD)
        result = await runner._handle_thread_command(event)
        assert "telegram" in result.lower()

    @pytest.mark.asyncio
    async def test_create_permission_error(self, tmp_path):
        """/thread <name> when bot lacks permissions shows helpful message."""
        bot = _make_mock_bot()
        bot.create_forum_topic = AsyncMock(
            side_effect=Exception("Not enough rights to manage topics")
        )
        runner = _make_runner(tmp_path, bot=bot)

        event = _make_event(text="/thread research")
        result = await runner._handle_thread_command(event)
        assert "permission" in result.lower() or "rights" in result.lower()

    @pytest.mark.asyncio
    async def test_threads_isolated_per_chat(self, tmp_path):
        """Threads in different chats are independent."""
        bot = _make_mock_bot()
        runner = _make_runner(tmp_path, bot=bot)

        # Create threads in different chats
        event_a = _make_event(text="/thread alpha", chat_id="111")
        event_b = _make_event(text="/thread beta", chat_id="222")
        await runner._handle_thread_command(event_a)
        await runner._handle_thread_command(event_b)

        # List for chat 111 should only have alpha
        list_a = await runner._handle_thread_command(
            _make_event(text="/thread list", chat_id="111")
        )
        assert "alpha" in list_a
        assert "beta" not in list_a

    @pytest.mark.asyncio
    async def test_close_api_error(self, tmp_path):
        """/thread close shows error if API call fails."""
        bot = _make_mock_bot()
        bot.close_forum_topic = AsyncMock(side_effect=Exception("API error"))
        runner = _make_runner(tmp_path, bot=bot)
        runner._thread_store.add("67890", "research", 42)

        event = _make_event(text="/thread close research")
        result = await runner._handle_thread_command(event)
        assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_bot_not_connected(self, tmp_path):
        """/thread when bot is not connected returns error."""
        runner = _make_runner(tmp_path)
        runner.adapters = {}  # No adapters

        event = _make_event(text="/thread research")
        result = await runner._handle_thread_command(event)
        assert "not connected" in result.lower()


# ---------------------------------------------------------------------------
# Command registry integration
# ---------------------------------------------------------------------------


class TestThreadCommandRegistry:
    """Verify /thread is properly registered in the command system."""

    @pytest.fixture(autouse=True)
    def _require_prompt_toolkit(self):
        pytest.importorskip("prompt_toolkit")

    def test_thread_in_registry(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("thread")
        assert cmd is not None
        assert cmd.name == "thread"
        assert cmd.gateway_only is True

    def test_thread_in_gateway_known_commands(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "thread" in GATEWAY_KNOWN_COMMANDS

    def test_thread_subcommands(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("thread")
        assert "list" in cmd.subcommands
        assert "close" in cmd.subcommands
