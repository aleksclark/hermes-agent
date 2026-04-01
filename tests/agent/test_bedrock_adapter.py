"""Property-based tests for agent.bedrock_adapter model metadata and ID resolution.

Uses the hypothesis library to verify correctness properties from the
AWS Bedrock Provider design document.
"""

import json

import hypothesis.strategies as st
from hypothesis import given, settings

from agent.bedrock_adapter import (
    BEDROCK_MODEL_ALIASES,
    BEDROCK_MODEL_METADATA,
    _convert_messages_to_bedrock,
    _convert_tools_to_bedrock,
    _is_cache_supported_model,
    _map_stop_reason,
    _strip_unsupported_schema_fields,
    build_bedrock_kwargs,
    get_bedrock_context_length,
    get_bedrock_max_output_tokens,
    get_bedrock_model_id,
    normalize_bedrock_response,
)


# ---------------------------------------------------------------------------
# Property 9: Model ID resolution handles all formats
# Feature: aws-bedrock-provider, Property 9: Model ID resolution handles all formats
# Validates: Requirements 6.1, 6.2, 6.4
# ---------------------------------------------------------------------------


class TestModelIdResolutionProperty:
    """Property 9 — get_bedrock_model_id resolves aliases, strips prefix, preserves cross-region."""

    @given(alias=st.sampled_from(sorted(BEDROCK_MODEL_ALIASES.keys())))
    @settings(max_examples=100)
    def test_alias_resolves_to_correct_full_id(self, alias: str) -> None:
        """Aliases (without prefix) resolve to the mapped full Bedrock model ID."""
        result = get_bedrock_model_id(alias)
        assert result == BEDROCK_MODEL_ALIASES[alias]

    @given(alias=st.sampled_from(sorted(BEDROCK_MODEL_ALIASES.keys())))
    @settings(max_examples=100)
    def test_alias_with_bedrock_prefix_resolves(self, alias: str) -> None:
        """Aliases with bedrock/ prefix resolve identically to bare aliases."""
        result = get_bedrock_model_id(f"bedrock/{alias}")
        assert result == BEDROCK_MODEL_ALIASES[alias]

    @given(full_id=st.sampled_from(sorted(BEDROCK_MODEL_METADATA.keys())))
    @settings(max_examples=100)
    def test_full_id_with_prefix_stripped(self, full_id: str) -> None:
        """Full model IDs with bedrock/ prefix pass through with prefix stripped."""
        result = get_bedrock_model_id(f"bedrock/{full_id}")
        assert result == full_id

    @given(full_id=st.sampled_from(sorted(BEDROCK_MODEL_METADATA.keys())))
    @settings(max_examples=100)
    def test_full_id_without_prefix_passes_through(self, full_id: str) -> None:
        """Full model IDs without bedrock/ prefix pass through unchanged."""
        result = get_bedrock_model_id(full_id)
        assert result == full_id

    @given(
        full_id=st.sampled_from(
            [k for k in sorted(BEDROCK_MODEL_METADATA.keys()) if k.startswith(("us.", "eu."))]
        )
    )
    @settings(max_examples=100)
    def test_cross_region_prefix_preserved(self, full_id: str) -> None:
        """Cross-region prefixes (us., eu.) are preserved in the resolved ID."""
        result = get_bedrock_model_id(f"bedrock/{full_id}")
        assert result == full_id
        assert result.startswith(("us.", "eu."))


# ---------------------------------------------------------------------------
# Property 4: Max tokens lookup returns correct value or fallback
# Feature: aws-bedrock-provider, Property 4: Max tokens lookup returns correct value or fallback
# Validates: Requirements 3.6
# ---------------------------------------------------------------------------

_known_models = sorted(BEDROCK_MODEL_METADATA.keys())


class TestMaxTokensLookupProperty:
    """Property 4 — get_bedrock_max_output_tokens returns metadata value or 8192 fallback."""

    @given(model_id=st.sampled_from(_known_models))
    @settings(max_examples=100)
    def test_known_model_returns_metadata_value(self, model_id: str) -> None:
        """Known models return the max_output_tokens value from BEDROCK_MODEL_METADATA."""
        result = get_bedrock_max_output_tokens(model_id)
        expected = BEDROCK_MODEL_METADATA[model_id]["max_output_tokens"]
        assert result == expected

    @given(
        model_id=st.one_of(
            st.sampled_from(_known_models),
            st.text(min_size=1).filter(lambda s: s not in BEDROCK_MODEL_METADATA),
        )
    )
    @settings(max_examples=100)
    def test_known_or_unknown_returns_correct_value(self, model_id: str) -> None:
        """Known models return metadata value; unknown models return 8192 fallback."""
        result = get_bedrock_max_output_tokens(model_id)
        if model_id in BEDROCK_MODEL_METADATA:
            assert result == BEDROCK_MODEL_METADATA[model_id]["max_output_tokens"]
        else:
            assert result == 8192


# ---------------------------------------------------------------------------
# Property 10: Context length metadata returns positive integers for known models
# Feature: aws-bedrock-provider, Property 10: Context length metadata returns positive integers for known models
# Validates: Requirements 6.3, 12.1
# ---------------------------------------------------------------------------


class TestContextLengthMetadataProperty:
    """Property 10 — get_bedrock_context_length returns positive int matching metadata table."""

    @given(model_id=st.sampled_from(sorted(BEDROCK_MODEL_METADATA.keys())))
    @settings(max_examples=100)
    def test_known_model_returns_positive_integer(self, model_id: str) -> None:
        """All known model IDs return a positive integer context length."""
        result = get_bedrock_context_length(model_id)
        assert isinstance(result, int)
        assert result > 0

    @given(model_id=st.sampled_from(sorted(BEDROCK_MODEL_METADATA.keys())))
    @settings(max_examples=100)
    def test_known_model_matches_metadata_table(self, model_id: str) -> None:
        """The returned context length matches the value in BEDROCK_MODEL_METADATA."""
        result = get_bedrock_context_length(model_id)
        expected = BEDROCK_MODEL_METADATA[model_id]["context_length"]
        assert result == expected


# ---------------------------------------------------------------------------
# Property 1: Message transformation round trip preserves structure
# Feature: aws-bedrock-provider, Property 1: Message transformation round trip preserves structure
# Validates: Requirements 3.1, 3.3, 3.4, 3.7
# ---------------------------------------------------------------------------

# Strategy helpers for generating OpenAI-format messages

_TOOL_NAMES = ["read_file", "write_file", "search", "execute", "list_dir"]


def _make_system_message(draw):
    text = draw(st.text(min_size=1, max_size=50))
    return {"role": "system", "content": text}


def _make_user_message(draw):
    text = draw(st.text(min_size=1, max_size=50))
    return {"role": "user", "content": text}


def _make_assistant_message_plain(draw):
    text = draw(st.text(min_size=1, max_size=50))
    return {"role": "assistant", "content": text}


def _make_assistant_message_with_tool_calls(draw):
    """Assistant message with 1-3 tool_calls."""
    n = draw(st.integers(min_value=1, max_value=3))
    tool_calls = []
    for i in range(n):
        name = draw(st.sampled_from(_TOOL_NAMES))
        tc = {
            "id": f"call_{i}_{draw(st.integers(min_value=0, max_value=9999))}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"path": f"file_{i}.txt"}),
            },
        }
        tool_calls.append(tc)
    return {
        "role": "assistant",
        "content": draw(st.one_of(st.just(""), st.text(min_size=1, max_size=30))),
        "tool_calls": tool_calls,
    }


def _make_tool_message(draw, tool_call_id: str):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": draw(st.text(min_size=1, max_size=50)),
    }


class TestMessageTransformationProperty:
    """Property 1 — _convert_messages_to_bedrock preserves message structure."""

    @given(data=st.data())
    @settings(max_examples=100)
    def test_system_messages_extracted_to_system_blocks(self, data) -> None:
        """System messages are extracted into system_blocks, not bedrock_messages."""
        n_system = data.draw(st.integers(min_value=1, max_value=3))
        messages = []
        for _ in range(n_system):
            messages.append(_make_system_message(data.draw))
        # Add at least one user message so there's a conversation
        messages.append(_make_user_message(data.draw))

        system_blocks, bedrock_messages = _convert_messages_to_bedrock(messages)

        # (a) system_blocks contains only system message text
        assert len(system_blocks) == n_system
        for block in system_blocks:
            assert "text" in block
            assert isinstance(block["text"], str)

        # (b) bedrock_messages has only "user" and "assistant" roles
        for msg in bedrock_messages:
            assert msg["role"] in ("user", "assistant")

    @given(data=st.data())
    @settings(max_examples=100)
    def test_bedrock_messages_only_user_and_assistant_roles(self, data) -> None:
        """All bedrock_messages have role 'user' or 'assistant' regardless of input roles."""
        messages = []
        # Mix of system, user, assistant, tool messages
        messages.append(_make_system_message(data.draw))
        messages.append(_make_user_message(data.draw))
        assistant_with_tc = _make_assistant_message_with_tool_calls(data.draw)
        messages.append(assistant_with_tc)
        # Add tool results for each tool_call
        for tc in assistant_with_tc["tool_calls"]:
            messages.append(_make_tool_message(data.draw, tc["id"]))
        messages.append(_make_user_message(data.draw))

        _, bedrock_messages = _convert_messages_to_bedrock(messages)

        for msg in bedrock_messages:
            assert msg["role"] in ("user", "assistant")

    @given(data=st.data())
    @settings(max_examples=100)
    def test_tool_calls_converted_to_tooluse_blocks(self, data) -> None:
        """Assistant tool_calls are converted to toolUse content blocks."""
        messages = [_make_user_message(data.draw)]
        assistant_msg = _make_assistant_message_with_tool_calls(data.draw)
        messages.append(assistant_msg)

        _, bedrock_messages = _convert_messages_to_bedrock(messages)

        # Find the assistant message in bedrock output
        assistant_msgs = [m for m in bedrock_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1

        # Check toolUse blocks exist with correct fields
        tool_use_blocks = []
        for amsg in assistant_msgs:
            for block in amsg["content"]:
                if "toolUse" in block:
                    tool_use_blocks.append(block["toolUse"])

        assert len(tool_use_blocks) == len(assistant_msg["tool_calls"])
        for tu in tool_use_blocks:
            assert "toolUseId" in tu
            assert "name" in tu
            assert "input" in tu

    @given(data=st.data())
    @settings(max_examples=100)
    def test_tool_messages_converted_to_toolresult_blocks(self, data) -> None:
        """Tool role messages are converted to toolResult content blocks."""
        messages = [_make_user_message(data.draw)]
        assistant_msg = _make_assistant_message_with_tool_calls(data.draw)
        messages.append(assistant_msg)
        tool_call_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        for tc_id in tool_call_ids:
            messages.append(_make_tool_message(data.draw, tc_id))

        _, bedrock_messages = _convert_messages_to_bedrock(messages)

        # Collect all toolResult blocks from user messages
        tool_result_blocks = []
        for msg in bedrock_messages:
            for block in msg["content"]:
                if "toolResult" in block:
                    tool_result_blocks.append(block["toolResult"])

        assert len(tool_result_blocks) == len(tool_call_ids)
        result_ids = {tr["toolUseId"] for tr in tool_result_blocks}
        assert result_ids == set(tool_call_ids)


# ---------------------------------------------------------------------------
# Property 2: Tool schema conversion produces valid Bedrock toolSpec
# Feature: aws-bedrock-provider, Property 2: Tool schema conversion produces valid Bedrock toolSpec and strips unsupported fields
# Validates: Requirements 3.2, 9.1, 9.4
# ---------------------------------------------------------------------------

# Fields that Bedrock does not support in JSON Schema
_UNSUPPORTED_FIELDS = {"additionalProperties", "$schema", "$ref", "$defs"}


def _schema_contains_unsupported(schema: dict) -> bool:
    """Recursively check if a schema dict contains any unsupported fields."""
    if not isinstance(schema, dict):
        return False
    for key, value in schema.items():
        if key in _UNSUPPORTED_FIELDS:
            return True
        if isinstance(value, dict) and _schema_contains_unsupported(value):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _schema_contains_unsupported(item):
                    return True
    return False


# Strategy for generating tool parameter schemas with possible unsupported fields
_property_schema = st.fixed_dictionaries(
    {"type": st.just("string")},
    optional={
        "description": st.text(min_size=1, max_size=30),
        "additionalProperties": st.booleans(),
        "$schema": st.just("http://json-schema.org/draft-07/schema#"),
    },
)

_parameters_schema = st.fixed_dictionaries(
    {
        "type": st.just("object"),
        "properties": st.dictionaries(
            keys=st.from_regex(r"[a-z_]{1,10}", fullmatch=True),
            values=_property_schema,
            min_size=1,
            max_size=4,
        ),
    },
    optional={
        "required": st.just(["param"]),
        "additionalProperties": st.booleans(),
        "$schema": st.just("http://json-schema.org/draft-07/schema#"),
        "$ref": st.just("#/definitions/Foo"),
        "$defs": st.just({"Foo": {"type": "string"}}),
    },
)

_tool_definition = st.fixed_dictionaries({
    "type": st.just("function"),
    "function": st.fixed_dictionaries({
        "name": st.from_regex(r"[a-z_]{1,15}", fullmatch=True),
        "description": st.text(min_size=1, max_size=50),
        "parameters": _parameters_schema,
    }),
})


class TestToolSchemaConversionProperty:
    """Property 2 — _convert_tools_to_bedrock produces valid toolSpec and strips unsupported fields."""

    @given(tools=st.lists(_tool_definition, min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_output_has_toolspec_structure(self, tools) -> None:
        """Each converted tool has toolSpec with name, description, inputSchema.json."""
        result = _convert_tools_to_bedrock(tools)

        assert len(result) == len(tools)
        for i, entry in enumerate(result):
            assert "toolSpec" in entry
            spec = entry["toolSpec"]
            assert "name" in spec
            assert "description" in spec
            assert "inputSchema" in spec
            assert "json" in spec["inputSchema"]
            # Name and description match input
            assert spec["name"] == tools[i]["function"]["name"]
            assert spec["description"] == tools[i]["function"]["description"]

    @given(tools=st.lists(_tool_definition, min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_output_has_no_unsupported_fields(self, tools) -> None:
        """Output schemas do not contain additionalProperties, $schema, $ref, or $defs."""
        result = _convert_tools_to_bedrock(tools)

        for entry in result:
            schema = entry["toolSpec"]["inputSchema"]["json"]
            assert not _schema_contains_unsupported(schema), (
                f"Unsupported fields found in output schema: {schema}"
            )

    @given(tools=st.lists(_tool_definition, min_size=1, max_size=3))
    @settings(max_examples=100)
    def test_strip_unsupported_schema_fields_removes_all(self, tools) -> None:
        """_strip_unsupported_schema_fields removes all unsupported fields recursively."""
        for tool in tools:
            params = tool["function"]["parameters"]
            stripped = _strip_unsupported_schema_fields(params)
            assert not _schema_contains_unsupported(stripped)


# ---------------------------------------------------------------------------
# Property 3: Inference parameter mapping preserves values
# Feature: aws-bedrock-provider, Property 3: Inference parameter mapping preserves values
# Validates: Requirements 3.5
# ---------------------------------------------------------------------------


class TestInferenceParameterMappingProperty:
    """Property 3 — build_bedrock_kwargs produces inferenceConfig with matching maxTokens."""

    @given(max_tokens=st.integers(min_value=1, max_value=100000))
    @settings(max_examples=100)
    def test_max_tokens_mapped_to_inference_config(self, max_tokens: int) -> None:
        """max_tokens is mapped to inferenceConfig.maxTokens with the same value."""
        messages = [{"role": "user", "content": "hello"}]
        result = build_bedrock_kwargs(
            model="amazon.nova-micro-v1:0",
            messages=messages,
            max_tokens=max_tokens,
        )

        assert "inferenceConfig" in result
        assert "maxTokens" in result["inferenceConfig"]
        assert result["inferenceConfig"]["maxTokens"] == max_tokens

    @given(max_tokens=st.integers(min_value=1, max_value=50000))
    @settings(max_examples=100)
    def test_max_tokens_positive_integer_preserved(self, max_tokens: int) -> None:
        """Any positive integer max_tokens is preserved exactly in the output."""
        messages = [{"role": "user", "content": "test"}]
        result = build_bedrock_kwargs(
            model="amazon.nova-micro-v1:0",
            messages=messages,
            max_tokens=max_tokens,
        )

        output_max = result["inferenceConfig"]["maxTokens"]
        assert isinstance(output_max, int)
        assert output_max == max_tokens


# ---------------------------------------------------------------------------
# Property 13: Consecutive same-role messages are merged or separated
# Feature: aws-bedrock-provider, Property 13: Consecutive same-role messages are merged or separated
# Validates: Requirements 9.3
# ---------------------------------------------------------------------------


class TestConsecutiveSameRoleMergingProperty:
    """Property 13 — _convert_messages_to_bedrock ensures no consecutive same-role messages."""

    @given(data=st.data())
    @settings(max_examples=100)
    def test_no_consecutive_same_role_in_output(self, data) -> None:
        """Output bedrock_messages never have two consecutive messages with the same role."""
        # Build a sequence with forced consecutive same-role messages
        n = data.draw(st.integers(min_value=2, max_value=8))
        messages = []
        for _ in range(n):
            role = data.draw(st.sampled_from(["user", "assistant"]))
            text = data.draw(st.text(min_size=1, max_size=30))
            msg = {"role": role, "content": text}
            messages.append(msg)

        _, bedrock_messages = _convert_messages_to_bedrock(messages)

        # Verify no two consecutive messages share the same role
        for i in range(1, len(bedrock_messages)):
            assert bedrock_messages[i]["role"] != bedrock_messages[i - 1]["role"], (
                f"Consecutive same-role at index {i}: "
                f"{bedrock_messages[i - 1]['role']} == {bedrock_messages[i]['role']}"
            )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_forced_consecutive_user_messages_merged(self, data) -> None:
        """Multiple consecutive user messages are merged into a single user message."""
        n_user = data.draw(st.integers(min_value=2, max_value=5))
        messages = []
        for _ in range(n_user):
            text = data.draw(st.text(min_size=1, max_size=20))
            messages.append({"role": "user", "content": text})

        _, bedrock_messages = _convert_messages_to_bedrock(messages)

        # All user content should be merged into one message
        user_msgs = [m for m in bedrock_messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        # The merged message should have content blocks from all originals
        assert len(user_msgs[0]["content"]) == n_user

    @given(data=st.data())
    @settings(max_examples=100)
    def test_forced_consecutive_assistant_messages_merged(self, data) -> None:
        """Multiple consecutive assistant messages are merged into a single assistant message."""
        messages = [{"role": "user", "content": "start"}]
        n_asst = data.draw(st.integers(min_value=2, max_value=5))
        for _ in range(n_asst):
            text = data.draw(st.text(min_size=1, max_size=20))
            messages.append({"role": "assistant", "content": text})

        _, bedrock_messages = _convert_messages_to_bedrock(messages)

        asst_msgs = [m for m in bedrock_messages if m["role"] == "assistant"]
        assert len(asst_msgs) == 1
        assert len(asst_msgs[0]["content"]) == n_asst


# ---------------------------------------------------------------------------
# Property 15: Prompt cache markers present only for supported models
# Feature: aws-bedrock-provider, Property 15: Prompt cache markers present only for supported models
# Validates: Requirements 10.1, 10.3
# ---------------------------------------------------------------------------

# Model IDs that contain "anthropic." are cache-supported
_CACHE_SUPPORTED_MODELS = [
    mid for mid in sorted(BEDROCK_MODEL_METADATA.keys()) if "anthropic." in mid
]
_CACHE_UNSUPPORTED_MODELS = [
    mid for mid in sorted(BEDROCK_MODEL_METADATA.keys()) if "anthropic." not in mid
]


def _has_cache_point(obj) -> bool:
    """Check if any part of the request kwargs contains a cachePoint marker."""
    if isinstance(obj, dict):
        if "cachePoint" in obj:
            return True
        return any(_has_cache_point(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_cache_point(item) for item in obj)
    return False


class TestPromptCacheMarkersProperty:
    """Property 15 — cache markers present iff model contains 'anthropic.'."""

    @given(model_id=st.sampled_from(_CACHE_SUPPORTED_MODELS))
    @settings(max_examples=100)
    def test_cache_markers_present_for_supported_models(self, model_id: str) -> None:
        """Cache-supported models (Claude/Anthropic) have cachePoint markers in output."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        result = build_bedrock_kwargs(model=model_id, messages=messages)

        assert _is_cache_supported_model(model_id)
        assert _has_cache_point(result), (
            f"Expected cache markers for supported model {model_id}"
        )

    @given(model_id=st.sampled_from(_CACHE_UNSUPPORTED_MODELS))
    @settings(max_examples=100)
    def test_no_cache_markers_for_unsupported_models(self, model_id: str) -> None:
        """Non-Anthropic models do not have cachePoint markers in output."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        result = build_bedrock_kwargs(model=model_id, messages=messages)

        assert not _is_cache_supported_model(model_id)
        assert not _has_cache_point(result), (
            f"Unexpected cache markers for unsupported model {model_id}"
        )

    @given(
        model_id=st.one_of(
            st.sampled_from(_CACHE_SUPPORTED_MODELS),
            st.sampled_from(_CACHE_UNSUPPORTED_MODELS),
        )
    )
    @settings(max_examples=100)
    def test_cache_marker_iff_anthropic_in_model_id(self, model_id: str) -> None:
        """Cache markers are present if and only if 'anthropic.' is in the model ID."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        result = build_bedrock_kwargs(model=model_id, messages=messages)

        has_cache = _has_cache_point(result)
        is_supported = "anthropic." in model_id
        assert has_cache == is_supported, (
            f"model={model_id}, has_cache={has_cache}, is_supported={is_supported}"
        )


# ---------------------------------------------------------------------------
# Property 5: Response transformation produces valid OpenAI-compatible structure
# Feature: aws-bedrock-provider, Property 5: Response transformation produces valid OpenAI-compatible structure
# Validates: Requirements 4.1, 4.2, 4.4, 9.2
# ---------------------------------------------------------------------------

# --- Hypothesis strategies for generating Bedrock response dicts ---

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

_bedrock_content_block = st.one_of(
    _bedrock_text_block,
    _bedrock_tool_use_block,
    _bedrock_reasoning_block,
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

_bedrock_response = st.builds(
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
    blocks=st.lists(_bedrock_content_block, min_size=1, max_size=6),
    stop=st.sampled_from(_STOP_REASONS),
    usage=_bedrock_usage,
)


class TestResponseTransformationProperty:
    """Property 5 — normalize_bedrock_response produces valid OpenAI-compatible structure."""

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_content_is_str_or_none(self, response: dict) -> None:
        """The .content attribute is a string or None."""
        msg, _ = normalize_bedrock_response(response)
        assert msg.content is None or isinstance(msg.content, str)

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_tool_calls_structure(self, response: dict) -> None:
        """The .tool_calls attribute is None or a list of properly shaped SimpleNamespace objects."""
        msg, _ = normalize_bedrock_response(response)
        if msg.tool_calls is not None:
            assert isinstance(msg.tool_calls, list)
            for tc in msg.tool_calls:
                # Each tool call has .id (str), .type == "function", .function with .name and .arguments
                assert isinstance(tc.id, str)
                assert tc.type == "function"
                assert isinstance(tc.function.name, str)
                assert isinstance(tc.function.arguments, str)
                # arguments must be valid JSON
                parsed = json.loads(tc.function.arguments)
                assert isinstance(parsed, dict)

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_tool_calls_ids_are_strings(self, response: dict) -> None:
        """All tool_calls have string .id values (uniqueness is Bedrock's responsibility)."""
        msg, _ = normalize_bedrock_response(response)
        if msg.tool_calls is not None:
            for tc in msg.tool_calls:
                assert isinstance(tc.id, str)
                assert len(tc.id) > 0

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_reasoning_is_str_or_none(self, response: dict) -> None:
        """The .reasoning attribute is a string or None."""
        msg, _ = normalize_bedrock_response(response)
        assert msg.reasoning is None or isinstance(msg.reasoning, str)

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_finish_reason_is_valid(self, response: dict) -> None:
        """The finish_reason is one of the valid OpenAI finish_reason values."""
        _, finish_reason = normalize_bedrock_response(response)
        assert finish_reason in ("stop", "tool_calls", "length")

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_text_blocks_accumulated_into_content(self, response: dict) -> None:
        """All text blocks in the response are concatenated into .content."""
        msg, _ = normalize_bedrock_response(response)
        text_blocks = [
            b["text"]
            for b in response["output"]["message"]["content"]
            if "text" in b
        ]
        if text_blocks:
            assert msg.content == "".join(text_blocks)
        else:
            assert msg.content is None

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_tool_use_blocks_become_tool_calls(self, response: dict) -> None:
        """Each toolUse block in the response becomes a tool_call entry."""
        msg, _ = normalize_bedrock_response(response)
        tool_use_blocks = [
            b["toolUse"]
            for b in response["output"]["message"]["content"]
            if "toolUse" in b
        ]
        if tool_use_blocks:
            assert msg.tool_calls is not None
            assert len(msg.tool_calls) == len(tool_use_blocks)
            for tc, tu in zip(msg.tool_calls, tool_use_blocks):
                assert tc.id == tu["toolUseId"]
                assert tc.function.name == tu["name"]
        else:
            assert msg.tool_calls is None

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_reasoning_blocks_accumulated(self, response: dict) -> None:
        """All reasoningContent blocks are concatenated into .reasoning."""
        msg, _ = normalize_bedrock_response(response)
        reasoning_texts = []
        for b in response["output"]["message"]["content"]:
            if "reasoningContent" in b:
                rt = b["reasoningContent"].get("reasoningText", {})
                t = rt.get("text", "")
                if t:
                    reasoning_texts.append(t)
        if reasoning_texts:
            assert msg.reasoning == "".join(reasoning_texts)
        else:
            assert msg.reasoning is None

    @given(response=_bedrock_response)
    @settings(max_examples=100)
    def test_usage_tokens_extracted(self, response: dict) -> None:
        """Token usage is extracted into .usage with correct field names."""
        msg, _ = normalize_bedrock_response(response)
        usage_data = response["usage"]
        assert msg.usage.prompt_tokens == usage_data["inputTokens"]
        assert msg.usage.completion_tokens == usage_data["outputTokens"]
        assert msg.usage.total_tokens == usage_data["totalTokens"]


# ---------------------------------------------------------------------------
# Property 6: Stop reason mapping is total over known values
# Feature: aws-bedrock-provider, Property 6: Stop reason mapping is total over known values
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------

_STOP_REASON_MAPPING = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


class TestStopReasonMappingProperty:
    """Property 6 — _map_stop_reason is total over all known Bedrock stopReason values."""

    @given(stop_reason=st.sampled_from(sorted(_STOP_REASON_MAPPING.keys())))
    @settings(max_examples=100)
    def test_known_stop_reasons_map_correctly(self, stop_reason: str) -> None:
        """Each known Bedrock stopReason maps to the correct OpenAI finish_reason."""
        result = _map_stop_reason(stop_reason)
        expected = _STOP_REASON_MAPPING[stop_reason]
        assert result == expected, (
            f"_map_stop_reason({stop_reason!r}) = {result!r}, expected {expected!r}"
        )

    @given(stop_reason=st.sampled_from(sorted(_STOP_REASON_MAPPING.keys())))
    @settings(max_examples=100)
    def test_result_is_valid_openai_finish_reason(self, stop_reason: str) -> None:
        """The mapped value is always a valid OpenAI finish_reason string."""
        result = _map_stop_reason(stop_reason)
        assert result in ("stop", "tool_calls", "length")

    def test_all_known_values_covered(self) -> None:
        """All four known Bedrock stopReason values produce a non-empty mapping."""
        for stop_reason, expected in _STOP_REASON_MAPPING.items():
            result = _map_stop_reason(stop_reason)
            assert result == expected

    @given(unknown=st.text(min_size=1).filter(lambda s: s not in _STOP_REASON_MAPPING))
    @settings(max_examples=100)
    def test_unknown_stop_reason_returns_default(self, unknown: str) -> None:
        """Unknown stopReason values fall back to 'stop'."""
        result = _map_stop_reason(unknown)
        assert result == "stop"


# ---------------------------------------------------------------------------
# Property 11: SigV4 signing produces valid Authorization header
# Feature: aws-bedrock-provider, Property 11: SigV4 signing produces valid Authorization header
# Validates: Requirements 7.1
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock, PropertyMock
from urllib.parse import quote as url_quote

from agent.bedrock_adapter import (
    _sign_request,
    BedrockCredentialResolver,
    BedrockAPIError,
    bedrock_converse_create,
)

# Strategy for generating AWS-style credentials
_aws_access_key = st.from_regex(r"AKIA[A-Z0-9]{16}", fullmatch=True)
_aws_secret_key = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/"),
    min_size=20,
    max_size=40,
)
_aws_region = st.sampled_from([
    "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1",
])


class TestSigV4SigningProperty:
    """Property 11 — _sign_request produces valid Authorization header via SigV4."""

    @given(
        body=st.binary(min_size=1, max_size=500),
        access_key=_aws_access_key,
        secret_key=_aws_secret_key,
        region=_aws_region,
    )
    @settings(max_examples=100)
    def test_sign_request_calls_sigv4auth_with_bedrock_service(
        self, body: bytes, access_key: str, secret_key: str, region: str
    ) -> None:
        """_sign_request calls SigV4Auth with 'bedrock' service name and the correct region."""
        import sys

        mock_creds_instance = MagicMock()
        mock_request_instance = MagicMock()
        mock_request_instance.headers = {"Authorization": "AWS4-HMAC-SHA256 ...", "Content-Type": "application/json"}

        mock_sigv4_cls = MagicMock(return_value=MagicMock())
        mock_creds_cls = MagicMock(return_value=mock_creds_instance)
        mock_awsreq_cls = MagicMock(return_value=mock_request_instance)

        # Build mock botocore modules
        mock_botocore_auth = MagicMock()
        mock_botocore_auth.SigV4Auth = mock_sigv4_cls
        mock_botocore_creds = MagicMock()
        mock_botocore_creds.Credentials = mock_creds_cls
        mock_botocore_awsreq = MagicMock()
        mock_botocore_awsreq.AWSRequest = mock_awsreq_cls

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = (access_key, secret_key, None)

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
        }):
            headers = _sign_request(
                url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
                body=body,
                credentials=resolver,
                region=region,
            )

            # Verify SigV4Auth was called with "bedrock" service and correct region
            mock_sigv4_cls.assert_called_once_with(mock_creds_instance, "bedrock", region)
            # Verify the returned headers contain Authorization
            assert "Authorization" in headers

    @given(
        body=st.binary(min_size=1, max_size=500),
        access_key=_aws_access_key,
        secret_key=_aws_secret_key,
        session_token=st.one_of(st.none(), st.text(min_size=10, max_size=40)),
        region=_aws_region,
    )
    @settings(max_examples=100)
    def test_sign_request_passes_credentials_to_botocore(
        self, body: bytes, access_key: str, secret_key: str, session_token, region: str
    ) -> None:
        """_sign_request passes the resolved credentials to botocore Credentials."""
        import sys

        mock_request_instance = MagicMock()
        mock_request_instance.headers = {"Authorization": "AWS4-HMAC-SHA256 Credential=..."}

        mock_creds_cls = MagicMock()
        mock_awsreq_cls = MagicMock(return_value=mock_request_instance)

        mock_botocore_auth = MagicMock()
        mock_botocore_auth.SigV4Auth = MagicMock(return_value=MagicMock())
        mock_botocore_creds = MagicMock()
        mock_botocore_creds.Credentials = mock_creds_cls
        mock_botocore_awsreq = MagicMock()
        mock_botocore_awsreq.AWSRequest = mock_awsreq_cls

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = (access_key, secret_key, session_token)

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
        }):
            _sign_request(
                url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
                body=body,
                credentials=resolver,
                region=region,
            )

            mock_creds_cls.assert_called_once_with(access_key, secret_key, session_token)


# ---------------------------------------------------------------------------
# Property 12: Endpoint URL construction follows Bedrock pattern
# Feature: aws-bedrock-provider, Property 12: Endpoint URL construction follows Bedrock pattern
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------

_bedrock_regions = st.sampled_from([
    "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
    "ap-southeast-1", "ap-northeast-1",
])

_bedrock_model_ids = st.sampled_from(sorted(BEDROCK_MODEL_METADATA.keys()))


class TestEndpointUrlConstructionProperty:
    """Property 12 — endpoint URL matches https://bedrock-runtime.{region}.amazonaws.com/model/{encoded_model_id}/converse."""

    @given(region=_bedrock_regions, model_id=_bedrock_model_ids)
    @settings(max_examples=100)
    def test_url_matches_bedrock_pattern(self, region: str, model_id: str) -> None:
        """The URL passed to httpx.post matches the expected Bedrock endpoint pattern."""
        import sys

        kwargs = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = ("AKIAIOSFODNN7EXAMPLE", "secret", None)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        mock_signed_headers = {"Authorization": "AWS4-HMAC-SHA256 ...", "Content-Type": "application/json"}

        # Mock botocore modules for _sign_request
        mock_botocore_auth = MagicMock()
        mock_botocore_creds = MagicMock()
        mock_botocore_awsreq = MagicMock()
        mock_aws_request = MagicMock()
        mock_aws_request.headers = mock_signed_headers
        mock_botocore_awsreq.AWSRequest.return_value = mock_aws_request

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
            "httpx": mock_httpx,
        }):
            # Remove AWS_BEDROCK_RUNTIME_ENDPOINT if set to ensure default URL is used
            import os as _os
            old_val = _os.environ.pop("AWS_BEDROCK_RUNTIME_ENDPOINT", None)
            try:
                bedrock_converse_create(kwargs, resolver, region)
            finally:
                if old_val is not None:
                    _os.environ["AWS_BEDROCK_RUNTIME_ENDPOINT"] = old_val

            called_url = mock_httpx.post.call_args[0][0]
            encoded_model_id = url_quote(model_id, safe="")
            expected_url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{encoded_model_id}/converse"
            assert called_url == expected_url

    @given(region=_bedrock_regions, model_id=_bedrock_model_ids)
    @settings(max_examples=100)
    def test_custom_endpoint_overrides_default(self, region: str, model_id: str) -> None:
        """When endpoint_url is provided, it replaces the default base URL."""
        import sys

        kwargs = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = ("AKIAIOSFODNN7EXAMPLE", "secret", None)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        }

        mock_signed_headers = {"Authorization": "AWS4-HMAC-SHA256 ..."}
        mock_botocore_auth = MagicMock()
        mock_botocore_creds = MagicMock()
        mock_botocore_awsreq = MagicMock()
        mock_aws_request = MagicMock()
        mock_aws_request.headers = mock_signed_headers
        mock_botocore_awsreq.AWSRequest.return_value = mock_aws_request

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response

        custom_endpoint = "https://vpce-abc123.bedrock-runtime.us-east-1.vpce.amazonaws.com"

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
            "httpx": mock_httpx,
        }):
            bedrock_converse_create(kwargs, resolver, region, endpoint_url=custom_endpoint)

            called_url = mock_httpx.post.call_args[0][0]
            encoded_model_id = url_quote(model_id, safe="")
            expected_url = f"{custom_endpoint}/model/{encoded_model_id}/converse"
            assert called_url == expected_url


# ---------------------------------------------------------------------------
# Property 14: Credential resolution respects priority order
# Feature: aws-bedrock-provider, Property 14: Credential resolution respects priority order
# Validates: Requirements 2.1, 2.2, 2.4
# ---------------------------------------------------------------------------


class TestCredentialPriorityProperty:
    """Property 14 — BedrockCredentialResolver uses highest-priority credential source."""

    @given(
        explicit_key=_aws_access_key,
        explicit_secret=_aws_secret_key,
        env_key=_aws_access_key,
        env_secret=_aws_secret_key,
    )
    @settings(max_examples=100)
    def test_explicit_credentials_override_env_vars(
        self, explicit_key: str, explicit_secret: str, env_key: str, env_secret: str
    ) -> None:
        """When explicit credentials are provided, they are used regardless of env vars."""
        import sys

        resolver = BedrockCredentialResolver(
            aws_access_key_id=explicit_key,
            aws_secret_access_key=explicit_secret,
        )

        # Mock boto3 so the import succeeds but Session is never called
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_frozen = MagicMock()
        mock_frozen.access_key = env_key
        mock_frozen.secret_key = env_secret
        mock_frozen.token = None
        mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen
        mock_boto3.Session.return_value = mock_session

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            access, secret, token = resolver.get_credentials()

            assert access == explicit_key
            assert secret == explicit_secret
            # boto3.Session should NOT have been called since explicit creds are used
            mock_boto3.Session.assert_not_called()

    @given(
        env_key=_aws_access_key,
        env_secret=_aws_secret_key,
    )
    @settings(max_examples=100)
    def test_boto3_session_used_when_no_explicit_credentials(
        self, env_key: str, env_secret: str
    ) -> None:
        """When no explicit credentials, boto3 Session is used to resolve them."""
        import sys

        resolver = BedrockCredentialResolver()
        # Clear any cached credentials from previous hypothesis examples
        resolver._cached_credentials = None
        resolver._cached_at = None

        mock_frozen = MagicMock()
        mock_frozen.access_key = env_key
        mock_frozen.secret_key = env_secret
        mock_frozen.token = None

        mock_creds = MagicMock()
        mock_creds.get_frozen_credentials.return_value = mock_frozen

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        mock_boto3 = MagicMock()
        mock_boto3.Session.return_value = mock_session

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            access, secret, token = resolver.get_credentials()

            assert access == env_key
            assert secret == env_secret
            mock_boto3.Session.assert_called_once()

    @given(
        explicit_key=_aws_access_key,
        explicit_secret=_aws_secret_key,
        session_token=st.text(min_size=10, max_size=40),
    )
    @settings(max_examples=100)
    def test_explicit_session_token_passed_through(
        self, explicit_key: str, explicit_secret: str, session_token: str
    ) -> None:
        """Explicit session token is returned alongside explicit key/secret."""
        import sys

        resolver = BedrockCredentialResolver(
            aws_access_key_id=explicit_key,
            aws_secret_access_key=explicit_secret,
            aws_session_token=session_token,
        )

        mock_boto3 = MagicMock()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            access, secret, token = resolver.get_credentials()

            assert access == explicit_key
            assert secret == explicit_secret
            assert token == session_token


# ---------------------------------------------------------------------------
# Property 16: Error responses produce descriptive exceptions
# Feature: aws-bedrock-provider, Property 16: Error responses produce descriptive exceptions
# Validates: Requirements 7.5, 11.5
# ---------------------------------------------------------------------------

_error_status_codes = st.sampled_from([400, 403, 404, 429, 500, 503])
_error_codes = st.sampled_from([
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
    "ThrottlingException",
    "InternalServerException",
    "ServiceUnavailableException",
])
_error_messages = st.text(min_size=5, max_size=100)
_request_ids = st.from_regex(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", fullmatch=True)


class TestErrorResponseProperty:
    """Property 16 — error responses from Bedrock produce descriptive BedrockAPIError exceptions."""

    @given(
        status_code=st.sampled_from([400, 404]),
        error_code=st.sampled_from(["ValidationException", "ResourceNotFoundException", "AccessDeniedException"]),
        error_message=_error_messages,
        request_id=_request_ids,
    )
    @settings(max_examples=100)
    def test_non_retryable_error_raises_with_status_and_message(
        self, status_code: int, error_code: str, error_message: str, request_id: str
    ) -> None:
        """Non-retryable errors raise BedrockAPIError containing the status code and error message."""
        import sys

        kwargs = {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = ("AKIAIOSFODNN7EXAMPLE", "secret", None)

        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.headers = {"x-amzn-requestid": request_id}
        mock_response.json.return_value = {
            "__type": error_code,
            "message": error_message,
        }
        mock_response.text = f'{{"__type": "{error_code}", "message": "{error_message}"}}'

        mock_signed_headers = {"Authorization": "AWS4-HMAC-SHA256 ..."}
        mock_botocore_auth = MagicMock()
        mock_botocore_creds = MagicMock()
        mock_botocore_awsreq = MagicMock()
        mock_aws_request = MagicMock()
        mock_aws_request.headers = mock_signed_headers
        mock_botocore_awsreq.AWSRequest.return_value = mock_aws_request

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
            "httpx": mock_httpx,
        }):
            try:
                bedrock_converse_create(kwargs, resolver, "us-east-1")
                assert False, "Expected BedrockAPIError to be raised"
            except BedrockAPIError as e:
                assert e.status_code == status_code
                assert error_code in str(e)
                assert error_message in str(e)

    @given(
        error_message=_error_messages,
        request_id=_request_ids,
    )
    @settings(max_examples=100)
    def test_validation_error_includes_details(
        self, error_message: str, request_id: str
    ) -> None:
        """ValidationException (400) includes the validation error details in the exception."""
        import sys

        kwargs = {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "test"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = ("AKIAIOSFODNN7EXAMPLE", "secret", None)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"x-amzn-requestid": request_id}
        mock_response.json.return_value = {
            "__type": "ValidationException",
            "message": error_message,
        }
        mock_response.text = json.dumps({"__type": "ValidationException", "message": error_message})

        mock_signed_headers = {"Authorization": "AWS4-HMAC-SHA256 ..."}
        mock_botocore_auth = MagicMock()
        mock_botocore_creds = MagicMock()
        mock_botocore_awsreq = MagicMock()
        mock_aws_request = MagicMock()
        mock_aws_request.headers = mock_signed_headers
        mock_botocore_awsreq.AWSRequest.return_value = mock_aws_request

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
            "httpx": mock_httpx,
        }):
            try:
                bedrock_converse_create(kwargs, resolver, "us-east-1")
                assert False, "Expected BedrockAPIError to be raised"
            except BedrockAPIError as e:
                assert e.status_code == 400
                assert "ValidationException" in str(e)
                assert error_message in str(e)
                assert e.request_id == request_id

    @given(
        status_code=st.sampled_from([500, 502]),
        error_message=_error_messages,
    )
    @settings(max_examples=100)
    def test_error_exception_contains_status_code_attribute(
        self, status_code: int, error_message: str
    ) -> None:
        """The raised BedrockAPIError has the status_code as an accessible attribute."""
        import sys

        kwargs = {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "test"}]}],
            "inferenceConfig": {"maxTokens": 100},
        }

        resolver = MagicMock(spec=BedrockCredentialResolver)
        resolver.get_credentials.return_value = ("AKIAIOSFODNN7EXAMPLE", "secret", None)

        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.headers = {"x-amzn-requestid": ""}
        mock_response.json.return_value = {
            "__type": "InternalServerException",
            "message": error_message,
        }
        mock_response.text = error_message

        mock_signed_headers = {"Authorization": "AWS4-HMAC-SHA256 ..."}
        mock_botocore_auth = MagicMock()
        mock_botocore_creds = MagicMock()
        mock_botocore_awsreq = MagicMock()
        mock_aws_request = MagicMock()
        mock_aws_request.headers = mock_signed_headers
        mock_botocore_awsreq.AWSRequest.return_value = mock_aws_request

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response

        with patch.dict(sys.modules, {
            "botocore": MagicMock(),
            "botocore.auth": mock_botocore_auth,
            "botocore.credentials": mock_botocore_creds,
            "botocore.awsrequest": mock_botocore_awsreq,
            "httpx": mock_httpx,
        }):
            try:
                bedrock_converse_create(kwargs, resolver, "us-east-1")
                assert False, "Expected BedrockAPIError to be raised"
            except BedrockAPIError as e:
                assert e.status_code == status_code
                assert isinstance(str(e), str)
                assert str(status_code) in str(e)


# ---------------------------------------------------------------------------
# Property 17: Compressed message history produces valid Bedrock requests
# Feature: aws-bedrock-provider, Property 17: Compressed message history produces valid Bedrock requests
# Validates: Requirements 12.3
# ---------------------------------------------------------------------------

# Strategy helpers for generating compressed message histories.
# A compressed history looks like:
#   1. A system message with a summary prefix (e.g. "[Context Summary] ...")
#   2. A few user/assistant message pairs (the truncated tail of the conversation)

_summary_prefix = st.sampled_from([
    "[Context Summary]",
    "[Conversation Summary]",
    "[Summary of previous context]",
    "[Compressed Context]",
])

_summary_body = st.text(min_size=10, max_size=200)


@st.composite
def compressed_message_history(draw):
    """Generate a message history that simulates context compression output.

    Shape:
    - 1 system message with a summary prefix
    - 1-4 user/assistant message pairs (truncated tail)
    """
    prefix = draw(_summary_prefix)
    body = draw(_summary_body)
    system_msg = {"role": "system", "content": f"{prefix} {body}"}

    n_pairs = draw(st.integers(min_value=1, max_value=4))
    conversation: list = []
    for _ in range(n_pairs):
        user_text = draw(st.text(min_size=1, max_size=80))
        assistant_text = draw(st.text(min_size=1, max_size=80))
        conversation.append({"role": "user", "content": user_text})
        conversation.append({"role": "assistant", "content": assistant_text})

    # Ensure the history ends with a user message (typical for a new turn)
    final_user = draw(st.text(min_size=1, max_size=80))
    conversation.append({"role": "user", "content": final_user})

    return [system_msg] + conversation


class TestCompressedMessageHistoryProperty:
    """Property 17 — compressed message histories produce valid Bedrock requests."""

    @given(messages=compressed_message_history())
    @settings(max_examples=100)
    def test_convert_messages_produces_valid_output(self, messages: list) -> None:
        """_convert_messages_to_bedrock succeeds on compressed histories and produces valid structure."""
        system_blocks, bedrock_messages = _convert_messages_to_bedrock(messages)

        # (a) system_blocks should contain the summary system message
        assert len(system_blocks) >= 1
        for block in system_blocks:
            assert "text" in block
            assert isinstance(block["text"], str)
            assert len(block["text"]) > 0

        # (b) bedrock_messages should only have user/assistant roles
        assert len(bedrock_messages) >= 1
        for msg in bedrock_messages:
            assert msg["role"] in ("user", "assistant")
            assert "content" in msg
            assert isinstance(msg["content"], list)
            assert len(msg["content"]) >= 1

        # (c) No consecutive same-role messages (Bedrock constraint)
        for i in range(1, len(bedrock_messages)):
            assert bedrock_messages[i]["role"] != bedrock_messages[i - 1]["role"]

    @given(messages=compressed_message_history())
    @settings(max_examples=100)
    def test_build_bedrock_kwargs_produces_valid_request(self, messages: list) -> None:
        """build_bedrock_kwargs succeeds on compressed histories and returns a valid request dict."""
        result = build_bedrock_kwargs(
            model="amazon.nova-micro-v1:0",
            messages=messages,
        )

        # Must have required top-level keys
        assert "modelId" in result
        assert "messages" in result
        assert "inferenceConfig" in result

        # modelId is a non-empty string
        assert isinstance(result["modelId"], str)
        assert len(result["modelId"]) > 0

        # messages is a non-empty list of dicts with role and content
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 1
        for msg in result["messages"]:
            assert "role" in msg
            assert msg["role"] in ("user", "assistant")
            assert "content" in msg

        # inferenceConfig has maxTokens
        assert "maxTokens" in result["inferenceConfig"]
        assert isinstance(result["inferenceConfig"]["maxTokens"], int)
        assert result["inferenceConfig"]["maxTokens"] > 0

        # system blocks should be present (from the summary system message)
        assert "system" in result
        assert isinstance(result["system"], list)
        assert len(result["system"]) >= 1

    @given(messages=compressed_message_history())
    @settings(max_examples=100)
    def test_summary_prefix_preserved_in_system_blocks(self, messages: list) -> None:
        """The summary prefix from the system message is preserved in the system blocks."""
        system_blocks, _ = _convert_messages_to_bedrock(messages)

        # The first system block should contain the summary text
        original_system_content = messages[0]["content"]
        combined_system_text = " ".join(b["text"] for b in system_blocks)
        # The original system content should appear in the combined system text
        assert original_system_content in combined_system_text or combined_system_text in original_system_content
