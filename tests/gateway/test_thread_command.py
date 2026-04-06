"""Tests for /thread gateway slash command.

Tests the _handle_thread_command handler (create, switch, list, delete
named virtual threads within a single chat) across gateway platforms.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(text="/thread", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(tmp_path):
    """Create a bare GatewayRunner with a real SessionStore backed by tmp_path."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()

    config = MagicMock(spec=GatewayConfig)
    config.sessions_dir = tmp_path / "sessions"
    config.group_sessions_per_user = True
    runner.config = config

    runner.session_store = SessionStore(
        sessions_dir=config.sessions_dir,
        config=config,
    )

    from gateway.session import ThreadStore
    runner._thread_store = ThreadStore(config.sessions_dir / "threads.json")

    return runner


# ---------------------------------------------------------------------------
# ThreadStore unit tests (persistence layer)
# ---------------------------------------------------------------------------


class TestThreadStore:
    """Tests for the ThreadStore persistence layer in gateway/session.py."""

    def test_import(self):
        from gateway.session import ThreadStore
        assert ThreadStore is not None

    def test_create_and_get_active(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:123", "research")
        assert store.get_active_thread("chat:123") == "research"

    def test_list_threads_empty(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        assert store.list_threads("chat:123") == []

    def test_list_threads(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:123", "alpha")
        store.set_active("chat:123", "beta")
        threads = store.list_threads("chat:123")
        assert "alpha" in threads
        assert "beta" in threads

    def test_delete_thread(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:123", "alpha")
        store.set_active("chat:123", "beta")
        store.delete_thread("chat:123", "alpha")
        assert "alpha" not in store.list_threads("chat:123")
        assert "beta" in store.list_threads("chat:123")

    def test_delete_active_resets_to_none(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:123", "research")
        assert store.get_active_thread("chat:123") == "research"
        store.delete_thread("chat:123", "research")
        assert store.get_active_thread("chat:123") is None

    def test_clear_active(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:123", "research")
        store.clear_active("chat:123")
        assert store.get_active_thread("chat:123") is None
        # Thread still exists
        assert "research" in store.list_threads("chat:123")

    def test_persistence_roundtrip(self, tmp_path):
        from gateway.session import ThreadStore
        path = tmp_path / "threads.json"

        store1 = ThreadStore(path)
        store1.set_active("chat:123", "alpha")
        store1.set_active("chat:123", "beta")

        store2 = ThreadStore(path)
        assert store2.list_threads("chat:123") == store1.list_threads("chat:123")
        assert store2.get_active_thread("chat:123") == "beta"

    def test_thread_names_are_case_preserved(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:123", "Research")
        assert store.get_active_thread("chat:123") == "Research"
        assert "Research" in store.list_threads("chat:123")

    def test_different_chats_isolated(self, tmp_path):
        from gateway.session import ThreadStore
        store = ThreadStore(tmp_path / "threads.json")

        store.set_active("chat:aaa", "alpha")
        store.set_active("chat:bbb", "beta")

        assert store.get_active_thread("chat:aaa") == "alpha"
        assert store.get_active_thread("chat:bbb") == "beta"
        assert store.list_threads("chat:aaa") == ["alpha"]
        assert store.list_threads("chat:bbb") == ["beta"]


# ---------------------------------------------------------------------------
# Session key construction for dynamic threads
# ---------------------------------------------------------------------------


class TestDynamicThreadSessionKey:
    """Verify that dynamic thread names produce correct session keys."""

    def test_dynamic_thread_key_format(self):
        """Dynamic thread session keys follow chat_id:thread:<name> pattern."""
        from gateway.session import dynamic_thread_session_key
        key = dynamic_thread_session_key("agent:main:telegram:dm:67890", "research")
        assert key == "agent:main:telegram:dm:67890:thread:research"

    def test_dynamic_thread_key_with_spaces(self):
        """Thread names with spaces are preserved in the key."""
        from gateway.session import dynamic_thread_session_key
        key = dynamic_thread_session_key("agent:main:telegram:dm:67890", "my research")
        assert key == "agent:main:telegram:dm:67890:thread:my research"


# ---------------------------------------------------------------------------
# /thread command handler tests
# ---------------------------------------------------------------------------


class TestHandleThreadCommand:
    """Tests for GatewayRunner._handle_thread_command."""

    @pytest.mark.asyncio
    async def test_no_args_shows_main(self, tmp_path):
        """/thread with no args shows 'main' when no thread is active."""
        runner = _make_runner(tmp_path)
        event = _make_event(text="/thread")
        result = await runner._handle_thread_command(event)
        assert "main" in result.lower()

    @pytest.mark.asyncio
    async def test_no_args_shows_active_thread(self, tmp_path):
        """/thread with no args shows the active thread name."""
        runner = _make_runner(tmp_path)

        # First create a thread
        create_event = _make_event(text="/thread research")
        await runner._handle_thread_command(create_event)

        # Now query current
        event = _make_event(text="/thread")
        result = await runner._handle_thread_command(event)
        assert "research" in result

    @pytest.mark.asyncio
    async def test_create_thread(self, tmp_path):
        """/thread <name> creates a new thread and switches to it."""
        runner = _make_runner(tmp_path)
        event = _make_event(text="/thread research")
        result = await runner._handle_thread_command(event)
        assert "research" in result

    @pytest.mark.asyncio
    async def test_switch_thread(self, tmp_path):
        """/thread <name> switches to an existing thread."""
        runner = _make_runner(tmp_path)

        # Create two threads
        await runner._handle_thread_command(_make_event(text="/thread alpha"))
        await runner._handle_thread_command(_make_event(text="/thread beta"))

        # Switch back to alpha
        result = await runner._handle_thread_command(_make_event(text="/thread alpha"))
        assert "alpha" in result

    @pytest.mark.asyncio
    async def test_switch_to_main(self, tmp_path):
        """/thread main returns to the default session."""
        runner = _make_runner(tmp_path)

        await runner._handle_thread_command(_make_event(text="/thread research"))
        result = await runner._handle_thread_command(_make_event(text="/thread main"))
        assert "main" in result.lower()

    @pytest.mark.asyncio
    async def test_list_threads(self, tmp_path):
        """/thread list shows all active threads."""
        runner = _make_runner(tmp_path)

        await runner._handle_thread_command(_make_event(text="/thread alpha"))
        await runner._handle_thread_command(_make_event(text="/thread beta"))

        result = await runner._handle_thread_command(_make_event(text="/thread list"))
        assert "alpha" in result
        assert "beta" in result

    @pytest.mark.asyncio
    async def test_list_threads_empty(self, tmp_path):
        """/thread list when no threads exist."""
        runner = _make_runner(tmp_path)
        result = await runner._handle_thread_command(_make_event(text="/thread list"))
        assert "no" in result.lower() or "none" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_thread(self, tmp_path):
        """/thread delete <name> removes a thread."""
        runner = _make_runner(tmp_path)

        await runner._handle_thread_command(_make_event(text="/thread research"))
        result = await runner._handle_thread_command(
            _make_event(text="/thread delete research")
        )
        assert "delete" in result.lower() or "removed" in result.lower()

        # Thread no longer in list
        list_result = await runner._handle_thread_command(
            _make_event(text="/thread list")
        )
        assert "research" not in list_result

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(self, tmp_path):
        """/thread delete <nonexistent> returns error."""
        runner = _make_runner(tmp_path)
        result = await runner._handle_thread_command(
            _make_event(text="/thread delete nonexistent")
        )
        assert "not found" in result.lower() or "no thread" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_main_rejected(self, tmp_path):
        """/thread delete main is rejected — cannot delete the default."""
        runner = _make_runner(tmp_path)
        result = await runner._handle_thread_command(
            _make_event(text="/thread delete main")
        )
        assert "cannot" in result.lower() or "can't" in result.lower()

    @pytest.mark.asyncio
    async def test_threads_isolated_per_chat(self, tmp_path):
        """Threads in different chats are independent."""
        runner = _make_runner(tmp_path)

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
    async def test_thread_sessions_independent(self, tmp_path):
        """Each thread gets its own session key."""
        runner = _make_runner(tmp_path)

        # Create a thread
        await runner._handle_thread_command(_make_event(text="/thread research"))

        # The thread store should have an active thread
        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="12345",
            chat_id="67890", user_name="testuser",
        )
        base_key = build_session_key(source)
        active = runner._thread_store.get_active_thread(base_key)
        assert active == "research"


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
        assert "delete" in cmd.subcommands
        assert "main" in cmd.subcommands
