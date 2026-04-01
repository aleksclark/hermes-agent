"""Property-based and unit tests for Bedrock streaming support.

Tests for:
- Property 7: Streaming events fire correct callbacks
- Property 8: Streaming assembly produces equivalent response to non-streaming
- Unit tests for streaming edge cases (fallback, thinking blocks, cache tokens)

Uses hypothesis for property tests and unittest.mock for mocking httpx/botocore.
"""

import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings, assume

from agent.bedrock_adapter import (
    BedrockAPIError,
    BedrockCredentialResolver,
    bedrock_converse_stream,
    bedrock_converse_create,
    normalize_bedrock_response,
)


# ---------------------------------------------------------------------------
# Shared helpers for mocking botocore / httpx in streaming tests
# ---------------------------------------------------------------------------

def _mock_botocore_modules():
    """Return a dict suitable for patch.dict(sys.modules, ...) that stubs botocore."""
    mock_botocore_auth = MagicMock()
    mock_botocore_creds = MagicMock()
    mock_botocore_awsreq = MagicMock()

    mock_aws_request = MagicMock()
    mock_aws_request.headers = {
        "Authorization": "AWS4-HMAC-SHA256 Credential=AKIA.../bedrock/aws4_request",
        "Content-Type": "application/json",
    }
    mock_botocore_awsreq.AWSRequest.return_value = mock_aws_request

    # EventStreamBuffer mock — will be configured per-test
    mock_eventstream = MagicMock()

    return {
        "botocore": MagicMock(),
        "botocore.auth": mock_botocore_auth,
        "botocore.credentials": mock_botocore_creds,
        "botocore.awsrequest": mock_botocore_awsreq,
        "botocore.eventstream": mock_eventstream,
    }


def _make_resolver():
    """Create a mock BedrockCredentialResolver."""
    resolver = MagicMock(spec=BedrockCredentialResolver)
    resolver.get_credentials.return_value = ("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/secret", None)
    return resolver


def _make_event_message(event_type: str, payload: dict, message_type: str = "event"):
    """Create a mock botocore EventStreamMessage."""
    msg = MagicMock()
    msg.headers = {
        ":event-type": event_type,
        ":message-type": message_type,
        ":content-type": "application/json",
    }
    msg.payload = json.dumps(payload).encode("utf-8") if payload else b""
    return msg


def _make_stream_event_messages(events: List[Tuple[str, dict]]):
    """Convert a list of (event_type, payload) tuples into mock EventStreamMessages."""
    return [_make_event_message(et, p) for et, p in events]


def _build_stream_events_for_text(text: str, block_index: int = 0) -> List[Tuple[str, dict]]:
    """Build stream events for a simple text content block."""
    return [
        ("contentBlockStart", {"contentBlockStart": {"contentBlockIndex": block_index, "start": {}}}),
        ("contentBlockDelta", {"contentBlockDelta": {"contentBlockIndex": block_index, "delta": {"text": text}}}),
        ("contentBlockStop", {"contentBlockStop": {"contentBlockIndex": block_index}}),
    ]


def _build_stream_events_for_tool(
    tool_name: str, tool_id: str, tool_input: dict, block_index: int = 0
) -> List[Tuple[str, dict]]:
    """Build stream events for a toolUse content block."""
    return [
        ("contentBlockStart", {"contentBlockStart": {
            "contentBlockIndex": block_index,
            "start": {"toolUse": {"toolUseId": tool_id, "name": tool_name}},
        }}),
        ("contentBlockDelta", {"contentBlockDelta": {
            "contentBlockIndex": block_index,
            "delta": {"toolUse": {"input": json.dumps(tool_input)}},
        }}),
        ("contentBlockStop", {"contentBlockStop": {"contentBlockIndex": block_index}}),
    ]


def _build_stream_events_for_reasoning(text: str, block_index: int = 0) -> List[Tuple[str, dict]]:
    """Build stream events for a reasoning/thinking content block."""
    return [
        ("contentBlockStart", {"contentBlockStart": {"contentBlockIndex": block_index, "start": {}}}),
        ("contentBlockDelta", {"contentBlockDelta": {
            "contentBlockIndex": block_index,
            "delta": {"reasoningContent": {"text": text}},
        }}),
        ("contentBlockStop", {"contentBlockStop": {"contentBlockIndex": block_index}}),
    ]


def _run_stream_with_events(
    events: List[Tuple[str, dict]],
    stream_delta_callback=None,
    reasoning_callback=None,
    tool_gen_callback=None,
):
    """Run bedrock_converse_stream with mocked httpx streaming and botocore EventStreamBuffer.

    Patches httpx.Client to return a mock streaming response, and patches
    botocore.eventstream.EventStreamBuffer to yield the given events.

    Returns the assembled response dict.
    """
    botocore_mods = _mock_botocore_modules()
    event_messages = _make_stream_event_messages(events)

    # Mock EventStreamBuffer: add_data is a no-op, __iter__ yields all messages
    mock_buf_instance = MagicMock()
    mock_buf_instance.add_data = MagicMock()
    mock_buf_instance.__iter__ = MagicMock(return_value=iter(event_messages))
    botocore_mods["botocore.eventstream"].EventStreamBuffer.return_value = mock_buf_instance

    # Mock httpx streaming response
    mock_stream_response = MagicMock()
    mock_stream_response.status_code = 200
    mock_stream_response.headers = {}
    # iter_bytes yields a single chunk (the buffer mock handles parsing)
    mock_stream_response.iter_bytes.return_value = [b"fake-binary-event-data"]
    mock_stream_response.__enter__ = MagicMock(return_value=mock_stream_response)
    mock_stream_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    mock_httpx = MagicMock()
    mock_httpx.Client.return_value = mock_client
    # Also mock the exception types used in the except clause
    mock_httpx.ConnectError = type("ConnectError", (Exception,), {})
    mock_httpx.ReadTimeout = type("ReadTimeout", (Exception,), {})
    mock_httpx.WriteTimeout = type("WriteTimeout", (Exception,), {})
    mock_httpx.PoolTimeout = type("PoolTimeout", (Exception,), {})
    mock_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
    mock_httpx.RemoteProtocolError = type("RemoteProtocolError", (Exception,), {})

    botocore_mods["httpx"] = mock_httpx

    kwargs = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "inferenceConfig": {"maxTokens": 100},
    }

    resolver = _make_resolver()

    with patch.dict(sys.modules, botocore_mods):
        result = bedrock_converse_stream(
            kwargs=kwargs,
            credentials=resolver,
            region="us-east-1",
            stream_delta_callback=stream_delta_callback,
            reasoning_callback=reasoning_callback,
            tool_gen_callback=tool_gen_callback,
        )

    return result


# ---------------------------------------------------------------------------
# Property 7: Streaming events fire correct callbacks
# Feature: aws-bedrock-provider, Property 7: Streaming events fire correct callbacks
# Validates: Requirements 5.1, 5.2, 5.3, 5.6
# ---------------------------------------------------------------------------

# Strategies for generating stream event sequences

_text_delta = st.text(min_size=1, max_size=50)
_tool_names = st.from_regex(r"[a-z_]{1,15}", fullmatch=True)
_tool_ids = st.from_regex(r"tooluse_[A-Za-z0-9]{6,12}", fullmatch=True)
_reasoning_text = st.text(min_size=1, max_size=50)


@st.composite
def stream_event_sequence(draw):
    """Generate a valid sequence of Bedrock stream events with mixed content types.

    Returns (events, expected_text_deltas, expected_tool_names, expected_reasoning_deltas).
    """
    events = [("messageStart", {"messageStart": {"role": "assistant"}})]
    expected_text_deltas = []
    expected_tool_names = []
    expected_reasoning_deltas = []

    block_index = 0
    # Generate 1-4 content blocks of random types
    n_blocks = draw(st.integers(min_value=1, max_value=4))
    for _ in range(n_blocks):
        block_type = draw(st.sampled_from(["text", "tool", "reasoning"]))

        if block_type == "text":
            text = draw(_text_delta)
            events.extend(_build_stream_events_for_text(text, block_index))
            expected_text_deltas.append(text)

        elif block_type == "tool":
            name = draw(_tool_names)
            tid = draw(_tool_ids)
            events.extend(_build_stream_events_for_tool(name, tid, {"key": "val"}, block_index))
            expected_tool_names.append(name)

        elif block_type == "reasoning":
            text = draw(_reasoning_text)
            events.extend(_build_stream_events_for_reasoning(text, block_index))
            expected_reasoning_deltas.append(text)

        block_index += 1

    stop = draw(st.sampled_from(["end_turn", "tool_use", "max_tokens"]))
    events.append(("messageStop", {"messageStop": {"stopReason": stop}}))

    inp_tokens = draw(st.integers(min_value=1, max_value=10000))
    out_tokens = draw(st.integers(min_value=1, max_value=10000))
    events.append(("metadata", {"metadata": {"usage": {
        "inputTokens": inp_tokens,
        "outputTokens": out_tokens,
        "totalTokens": inp_tokens + out_tokens,
    }}}))

    return events, expected_text_deltas, expected_tool_names, expected_reasoning_deltas


class TestStreamingCallbacksProperty:
    """Property 7 — Streaming events fire correct callbacks."""

    @given(data=stream_event_sequence())
    @settings(max_examples=100)
    def test_text_delta_callback_fired_for_text_deltas(self, data) -> None:
        """stream_delta_callback is called for every contentBlockDelta with a text delta."""
        events, expected_text_deltas, _, _ = data

        text_callback = MagicMock()
        _run_stream_with_events(events, stream_delta_callback=text_callback)

        actual_texts = [call.args[0] for call in text_callback.call_args_list]
        assert actual_texts == expected_text_deltas

    @given(data=stream_event_sequence())
    @settings(max_examples=100)
    def test_tool_gen_callback_fired_for_tool_starts(self, data) -> None:
        """tool_gen_callback is called for every contentBlockStart with a toolUse start."""
        events, _, expected_tool_names, _ = data

        tool_callback = MagicMock()
        _run_stream_with_events(events, tool_gen_callback=tool_callback)

        actual_names = [call.args[0] for call in tool_callback.call_args_list]
        assert actual_names == expected_tool_names

    @given(data=stream_event_sequence())
    @settings(max_examples=100)
    def test_reasoning_callback_fired_for_reasoning_deltas(self, data) -> None:
        """reasoning_callback is called for every contentBlockDelta with reasoningContent."""
        events, _, _, expected_reasoning = data

        reasoning_cb = MagicMock()
        _run_stream_with_events(events, reasoning_callback=reasoning_cb)

        actual_reasoning = [call.args[0] for call in reasoning_cb.call_args_list]
        assert actual_reasoning == expected_reasoning

    @given(data=stream_event_sequence())
    @settings(max_examples=100)
    def test_no_callbacks_when_none_provided(self, data) -> None:
        """When callbacks are None, streaming completes without errors."""
        events, _, _, _ = data
        # Should not raise
        result = _run_stream_with_events(events)
        assert "output" in result


# ---------------------------------------------------------------------------
# Property 8: Streaming assembly produces equivalent response to non-streaming
# Feature: aws-bedrock-provider, Property 8: Streaming assembly produces equivalent response to non-streaming
# Validates: Requirements 5.4, 5.5
# ---------------------------------------------------------------------------

# Strategies for generating complete Bedrock responses that we split into events

_STOP_REASONS = ["end_turn", "tool_use", "max_tokens", "stop_sequence"]

_bedrock_text_block = st.builds(
    lambda t: {"text": t},
    t=st.text(min_size=1, max_size=80),
)

_bedrock_tool_use_block = st.builds(
    lambda tid, name, inp: {
        "toolUse": {
            "toolUseId": tid,
            "name": name,
            "input": inp,
        }
    },
    tid=st.from_regex(r"tooluse_[A-Za-z0-9]{6,12}", fullmatch=True),
    name=st.from_regex(r"[a-z_]{1,15}", fullmatch=True),
    inp=st.fixed_dictionaries(
        {},
        optional={
            "path": st.text(min_size=1, max_size=20),
            "query": st.text(min_size=1, max_size=20),
        },
    ),
)

_bedrock_reasoning_block = st.builds(
    lambda t: {"reasoningContent": {"reasoningText": {"text": t}}},
    t=st.text(min_size=1, max_size=80),
)

_bedrock_usage = st.builds(
    lambda inp, out: {
        "inputTokens": inp,
        "outputTokens": out,
        "totalTokens": inp + out,
    },
    inp=st.integers(min_value=1, max_value=100000),
    out=st.integers(min_value=1, max_value=100000),
)

# Use only one type per response to keep the split-into-events logic simple
_single_type_content = st.one_of(
    st.lists(_bedrock_text_block, min_size=1, max_size=3),
    st.lists(_bedrock_tool_use_block, min_size=1, max_size=2),
    st.lists(_bedrock_reasoning_block, min_size=1, max_size=2),
)

_bedrock_response_for_streaming = st.builds(
    lambda blocks, stop, usage: {
        "output": {
            "message": {
                "role": "assistant",
                "content": blocks,
            }
        },
        "stopReason": stop,
        "usage": usage,
    },
    blocks=_single_type_content,
    stop=st.sampled_from(_STOP_REASONS),
    usage=_bedrock_usage,
)


def _split_response_into_events(response: dict) -> List[Tuple[str, dict]]:
    """Split a complete Bedrock response into a sequence of stream events."""
    events = [("messageStart", {"messageStart": {"role": response["output"]["message"]["role"]}})]

    for idx, block in enumerate(response["output"]["message"]["content"]):
        if "text" in block:
            events.extend(_build_stream_events_for_text(block["text"], idx))
        elif "toolUse" in block:
            tu = block["toolUse"]
            events.extend(_build_stream_events_for_tool(
                tu["name"], tu["toolUseId"], tu["input"], idx
            ))
        elif "reasoningContent" in block:
            text = block["reasoningContent"]["reasoningText"]["text"]
            events.extend(_build_stream_events_for_reasoning(text, idx))

    events.append(("messageStop", {"messageStop": {"stopReason": response["stopReason"]}}))
    events.append(("metadata", {"metadata": {"usage": response["usage"]}}))

    return events


class TestStreamingAssemblyEquivalenceProperty:
    """Property 8 — Streaming assembly produces equivalent response to non-streaming."""

    @given(response=_bedrock_response_for_streaming)
    @settings(max_examples=100)
    def test_assembled_response_has_correct_structure(self, response: dict) -> None:
        """The assembled streaming response has output.message.content, stopReason, and usage."""
        events = _split_response_into_events(response)
        assembled = _run_stream_with_events(events)

        assert "output" in assembled
        assert "message" in assembled["output"]
        assert "content" in assembled["output"]["message"]
        assert "stopReason" in assembled
        assert "usage" in assembled

    @given(response=_bedrock_response_for_streaming)
    @settings(max_examples=100)
    def test_assembled_stop_reason_matches(self, response: dict) -> None:
        """The assembled stopReason matches the original response."""
        events = _split_response_into_events(response)
        assembled = _run_stream_with_events(events)

        assert assembled["stopReason"] == response["stopReason"]

    @given(response=_bedrock_response_for_streaming)
    @settings(max_examples=100)
    def test_assembled_usage_matches(self, response: dict) -> None:
        """The assembled usage tokens match the original response."""
        events = _split_response_into_events(response)
        assembled = _run_stream_with_events(events)

        assert assembled["usage"]["inputTokens"] == response["usage"]["inputTokens"]
        assert assembled["usage"]["outputTokens"] == response["usage"]["outputTokens"]

    @given(response=_bedrock_response_for_streaming)
    @settings(max_examples=100)
    def test_normalized_output_structurally_equivalent(self, response: dict) -> None:
        """normalize_bedrock_response on the assembled response produces equivalent output."""
        events = _split_response_into_events(response)
        assembled = _run_stream_with_events(events)

        # Normalize both
        orig_msg, orig_reason = normalize_bedrock_response(response)
        assembled_msg, assembled_reason = normalize_bedrock_response(assembled)

        # finish_reason must match
        assert assembled_reason == orig_reason

        # content must match
        assert assembled_msg.content == orig_msg.content

        # reasoning must match
        assert assembled_msg.reasoning == orig_msg.reasoning

        # tool_calls structure must match
        if orig_msg.tool_calls is None:
            assert assembled_msg.tool_calls is None
        else:
            assert assembled_msg.tool_calls is not None
            assert len(assembled_msg.tool_calls) == len(orig_msg.tool_calls)
            for orig_tc, asm_tc in zip(orig_msg.tool_calls, assembled_msg.tool_calls):
                assert asm_tc.id == orig_tc.id
                assert asm_tc.function.name == orig_tc.function.name
                # Arguments should parse to the same dict
                assert json.loads(asm_tc.function.arguments) == json.loads(orig_tc.function.arguments)


# ---------------------------------------------------------------------------
# Unit tests for streaming edge cases (Task 8.4)
# Tests: fallback on connection failure, thinking block extraction,
#        cache token extraction from metadata events
# Validates: Requirements 5.6, 5.7, 10.2
# ---------------------------------------------------------------------------


class TestStreamFallbackOnConnectionFailure:
    """Test that streaming falls back to non-streaming on connection errors."""

    def test_fallback_on_connect_error(self) -> None:
        """When httpx raises ConnectError, bedrock_converse_stream falls back to bedrock_converse_create."""
        botocore_mods = _mock_botocore_modules()

        # Create real-ish exception classes for httpx
        class MockConnectError(Exception):
            pass

        mock_httpx = MagicMock()
        mock_httpx.ConnectError = MockConnectError
        mock_httpx.ReadTimeout = type("ReadTimeout", (Exception,), {})
        mock_httpx.WriteTimeout = type("WriteTimeout", (Exception,), {})
        mock_httpx.PoolTimeout = type("PoolTimeout", (Exception,), {})
        mock_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
        mock_httpx.RemoteProtocolError = type("RemoteProtocolError", (Exception,), {})

        # Make Client().stream() raise ConnectError
        mock_client = MagicMock()
        mock_client.stream.side_effect = MockConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        # Also mock the non-streaming post for the fallback path
        fallback_response = MagicMock()
        fallback_response.status_code = 200
        fallback_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "fallback response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        mock_httpx.post.return_value = fallback_response

        botocore_mods["httpx"] = mock_httpx

        kwargs = {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }
        resolver = _make_resolver()

        with patch.dict(sys.modules, botocore_mods):
            result = bedrock_converse_stream(
                kwargs=kwargs,
                credentials=resolver,
                region="us-east-1",
            )

        # The fallback should have called httpx.post (non-streaming)
        assert mock_httpx.post.called
        assert result["output"]["message"]["content"][0]["text"] == "fallback response"

    def test_fallback_on_read_timeout(self) -> None:
        """When httpx raises ReadTimeout during streaming, falls back to non-streaming."""
        botocore_mods = _mock_botocore_modules()

        class MockReadTimeout(Exception):
            pass

        mock_httpx = MagicMock()
        mock_httpx.ConnectError = type("ConnectError", (Exception,), {})
        mock_httpx.ReadTimeout = MockReadTimeout
        mock_httpx.WriteTimeout = type("WriteTimeout", (Exception,), {})
        mock_httpx.PoolTimeout = type("PoolTimeout", (Exception,), {})
        mock_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
        mock_httpx.RemoteProtocolError = type("RemoteProtocolError", (Exception,), {})

        # Make the stream context manager's __enter__ work, but iter_bytes raises
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.headers = {}
        mock_stream_response.iter_bytes.side_effect = MockReadTimeout("Read timed out")
        mock_stream_response.__enter__ = MagicMock(return_value=mock_stream_response)
        mock_stream_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        fallback_response = MagicMock()
        fallback_response.status_code = 200
        fallback_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "timeout fallback"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 8, "outputTokens": 3, "totalTokens": 11},
        }
        mock_httpx.post.return_value = fallback_response

        botocore_mods["httpx"] = mock_httpx

        # Also need EventStreamBuffer for the _iter_stream_events path
        mock_buf = MagicMock()
        mock_buf.add_data.side_effect = MockReadTimeout("Read timed out")
        botocore_mods["botocore.eventstream"].EventStreamBuffer.return_value = mock_buf

        kwargs = {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }
        resolver = _make_resolver()

        with patch.dict(sys.modules, botocore_mods):
            result = bedrock_converse_stream(
                kwargs=kwargs,
                credentials=resolver,
                region="us-east-1",
            )

        assert mock_httpx.post.called
        assert result["output"]["message"]["content"][0]["text"] == "timeout fallback"


class TestThinkingBlockExtraction:
    """Test that thinking/reasoning blocks are correctly extracted from streaming events."""

    def test_single_thinking_block(self) -> None:
        """A single reasoning block is assembled correctly."""
        events = [
            ("messageStart", {"messageStart": {"role": "assistant"}}),
            *_build_stream_events_for_reasoning("Let me think about this...", block_index=0),
            *_build_stream_events_for_text("Here is my answer.", block_index=1),
            ("messageStop", {"messageStop": {"stopReason": "end_turn"}}),
            ("metadata", {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70}}}),
        ]

        result = _run_stream_with_events(events)
        msg, reason = normalize_bedrock_response(result)

        assert msg.reasoning == "Let me think about this..."
        assert msg.content == "Here is my answer."
        assert reason == "stop"

    def test_multi_chunk_thinking_block(self) -> None:
        """Reasoning text split across multiple deltas is concatenated."""
        events = [
            ("messageStart", {"messageStart": {"role": "assistant"}}),
            ("contentBlockStart", {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}),
            ("contentBlockDelta", {"contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "First part. "}},
            }}),
            ("contentBlockDelta", {"contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "Second part."}},
            }}),
            ("contentBlockStop", {"contentBlockStop": {"contentBlockIndex": 0}}),
            ("messageStop", {"messageStop": {"stopReason": "end_turn"}}),
            ("metadata", {"metadata": {"usage": {"inputTokens": 30, "outputTokens": 10, "totalTokens": 40}}}),
        ]

        reasoning_cb = MagicMock()
        result = _run_stream_with_events(events, reasoning_callback=reasoning_cb)
        msg, _ = normalize_bedrock_response(result)

        assert msg.reasoning == "First part. Second part."
        # Callback should have been called twice
        assert reasoning_cb.call_count == 2
        assert reasoning_cb.call_args_list[0].args[0] == "First part. "
        assert reasoning_cb.call_args_list[1].args[0] == "Second part."

    def test_thinking_with_tool_use(self) -> None:
        """Reasoning block followed by tool use block both assembled correctly."""
        events = [
            ("messageStart", {"messageStart": {"role": "assistant"}}),
            *_build_stream_events_for_reasoning("I need to read a file.", block_index=0),
            *_build_stream_events_for_tool("read_file", "tool_abc123", {"path": "test.py"}, block_index=1),
            ("messageStop", {"messageStop": {"stopReason": "tool_use"}}),
            ("metadata", {"metadata": {"usage": {"inputTokens": 40, "outputTokens": 15, "totalTokens": 55}}}),
        ]

        result = _run_stream_with_events(events)
        msg, reason = normalize_bedrock_response(result)

        assert msg.reasoning == "I need to read a file."
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "read_file"
        assert msg.tool_calls[0].id == "tool_abc123"
        assert reason == "tool_calls"


class TestCacheTokenExtraction:
    """Test that cache read/write token counts are extracted from metadata events."""

    def test_cache_tokens_in_metadata(self) -> None:
        """Cache token counts from metadata are preserved in the assembled response."""
        events = [
            ("messageStart", {"messageStart": {"role": "assistant"}}),
            *_build_stream_events_for_text("Cached response.", block_index=0),
            ("messageStop", {"messageStop": {"stopReason": "end_turn"}}),
            ("metadata", {"metadata": {"usage": {
                "inputTokens": 100,
                "outputTokens": 25,
                "totalTokens": 125,
                "cacheReadInputTokenCount": 80,
                "cacheWriteInputTokenCount": 20,
            }}}),
        ]

        result = _run_stream_with_events(events)

        # Verify the raw usage dict has cache fields
        assert result["usage"]["cacheReadInputTokenCount"] == 80
        assert result["usage"]["cacheWriteInputTokenCount"] == 20

        # Verify normalize_bedrock_response extracts them
        msg, _ = normalize_bedrock_response(result)
        assert msg.usage.cache_read_input_tokens == 80
        assert msg.usage.cache_creation_input_tokens == 20

    def test_no_cache_tokens_when_absent(self) -> None:
        """When metadata has no cache fields, usage still works correctly."""
        events = [
            ("messageStart", {"messageStart": {"role": "assistant"}}),
            *_build_stream_events_for_text("No cache.", block_index=0),
            ("messageStop", {"messageStop": {"stopReason": "end_turn"}}),
            ("metadata", {"metadata": {"usage": {
                "inputTokens": 50,
                "outputTokens": 10,
                "totalTokens": 60,
            }}}),
        ]

        result = _run_stream_with_events(events)
        msg, _ = normalize_bedrock_response(result)

        assert msg.usage.prompt_tokens == 50
        assert msg.usage.completion_tokens == 10
        assert not hasattr(msg.usage, "cache_read_input_tokens")
        assert not hasattr(msg.usage, "cache_creation_input_tokens")

    def test_cache_tokens_with_zero_values(self) -> None:
        """Cache token counts of zero are still extracted."""
        events = [
            ("messageStart", {"messageStart": {"role": "assistant"}}),
            *_build_stream_events_for_text("Zero cache.", block_index=0),
            ("messageStop", {"messageStop": {"stopReason": "end_turn"}}),
            ("metadata", {"metadata": {"usage": {
                "inputTokens": 30,
                "outputTokens": 5,
                "totalTokens": 35,
                "cacheReadInputTokenCount": 0,
                "cacheWriteInputTokenCount": 0,
            }}}),
        ]

        result = _run_stream_with_events(events)
        msg, _ = normalize_bedrock_response(result)

        assert msg.usage.cache_read_input_tokens == 0
        assert msg.usage.cache_creation_input_tokens == 0
