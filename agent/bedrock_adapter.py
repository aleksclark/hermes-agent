"""AWS Bedrock Converse API adapter for Hermes Agent.

Provides model metadata, ID resolution, and utility functions for the
Bedrock provider. Follows the same adapter pattern as anthropic_adapter.py.

boto3 is lazy-imported — this module can be imported without boto3 installed.
"""

import json
import base64
import os
import random
import re
import time
from types import SimpleNamespace
from typing import Tuple, List, Dict, Any, Optional
from urllib.parse import quote as url_quote

# ---------------------------------------------------------------------------
# Static Model Metadata Table
# Maps known Bedrock model IDs to context_length and max_output_tokens.
# ---------------------------------------------------------------------------

BEDROCK_MODEL_METADATA: Dict[str, Dict[str, int]] = {
    # Anthropic Claude models
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"context_length": 200000, "max_output_tokens": 8192},
    "anthropic.claude-3-5-haiku-20241022-v1:0": {"context_length": 200000, "max_output_tokens": 8192},
    "anthropic.claude-3-opus-20240229-v1:0": {"context_length": 200000, "max_output_tokens": 4096},
    "anthropic.claude-3-sonnet-20240229-v1:0": {"context_length": 200000, "max_output_tokens": 4096},
    "anthropic.claude-3-haiku-20240307-v1:0": {"context_length": 200000, "max_output_tokens": 4096},
    # Claude 4 / Sonnet 4 / Opus 4 (cross-region)
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {"context_length": 200000, "max_output_tokens": 16384},
    "us.anthropic.claude-opus-4-20250514-v1:0": {"context_length": 200000, "max_output_tokens": 16384},
    "eu.anthropic.claude-sonnet-4-20250514-v1:0": {"context_length": 200000, "max_output_tokens": 16384},
    # Amazon Nova models
    "amazon.nova-pro-v1:0": {"context_length": 300000, "max_output_tokens": 5120},
    "amazon.nova-lite-v1:0": {"context_length": 300000, "max_output_tokens": 5120},
    "amazon.nova-micro-v1:0": {"context_length": 128000, "max_output_tokens": 5120},
    # Meta Llama models
    "meta.llama3-1-405b-instruct-v1:0": {"context_length": 128000, "max_output_tokens": 4096},
    "meta.llama3-1-70b-instruct-v1:0": {"context_length": 128000, "max_output_tokens": 4096},
    "meta.llama3-1-8b-instruct-v1:0": {"context_length": 128000, "max_output_tokens": 4096},
    # Mistral models
    "mistral.mistral-large-2407-v1:0": {"context_length": 128000, "max_output_tokens": 8192},
}

# ---------------------------------------------------------------------------
# Short aliases → full Bedrock model IDs
# ---------------------------------------------------------------------------

BEDROCK_MODEL_ALIASES: Dict[str, str] = {
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "nova-micro": "amazon.nova-micro-v1:0",
    "llama3.1-405b": "meta.llama3-1-405b-instruct-v1:0",
    "llama3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "mistral-large": "mistral.mistral-large-2407-v1:0",
}


# ---------------------------------------------------------------------------
# Model ID resolution
# ---------------------------------------------------------------------------

def get_bedrock_model_id(model: str) -> str:
    """Resolve a Hermes model string to a full Bedrock model ID.

    Strips the ``bedrock/`` prefix, resolves short aliases from
    BEDROCK_MODEL_ALIASES, and preserves cross-region prefixes
    (e.g. ``us.``, ``eu.``).

    Examples::

        get_bedrock_model_id("bedrock/claude-sonnet-4")
        # → "us.anthropic.claude-sonnet-4-20250514-v1:0"

        get_bedrock_model_id("bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")
        # → "us.anthropic.claude-sonnet-4-20250514-v1:0"

        get_bedrock_model_id("anthropic.claude-3-5-sonnet-20241022-v2:0")
        # → "anthropic.claude-3-5-sonnet-20241022-v2:0"
    """
    # Strip bedrock/ prefix if present
    if model.startswith("bedrock/"):
        model = model[len("bedrock/"):]

    # Resolve alias if it matches
    if model in BEDROCK_MODEL_ALIASES:
        return BEDROCK_MODEL_ALIASES[model]

    return model


def get_bedrock_context_length(model: str) -> int:
    """Return context length for a Bedrock model from the static metadata table.

    Resolves the model ID via :func:`get_bedrock_model_id`, then looks up
    the metadata. For cross-region models (e.g. ``us.anthropic.…``), tries
    both the full ID and the ID without the region prefix.

    Raises:
        KeyError: If the model is not found in BEDROCK_MODEL_METADATA.
    """
    model_id = get_bedrock_model_id(model)

    # Direct lookup
    if model_id in BEDROCK_MODEL_METADATA:
        return BEDROCK_MODEL_METADATA[model_id]["context_length"]

    # Try without cross-region prefix (e.g. strip "us." or "eu.")
    parts = model_id.split(".", 1)
    if len(parts) == 2 and len(parts[0]) <= 3:
        base_id = parts[1]
        if base_id in BEDROCK_MODEL_METADATA:
            return BEDROCK_MODEL_METADATA[base_id]["context_length"]

    raise KeyError(
        f"Unknown Bedrock model: {model_id!r}. "
        f"Known models: {', '.join(sorted(BEDROCK_MODEL_METADATA.keys()))}"
    )


def get_bedrock_max_output_tokens(model: str) -> int:
    """Return max output tokens for a Bedrock model.

    Resolves the model ID via :func:`get_bedrock_model_id`, then looks up
    the metadata. For cross-region models, tries both with and without the
    region prefix. Falls back to 8192 if the model is not found.
    """
    model_id = get_bedrock_model_id(model)

    # Direct lookup
    if model_id in BEDROCK_MODEL_METADATA:
        return BEDROCK_MODEL_METADATA[model_id]["max_output_tokens"]

    # Try without cross-region prefix
    parts = model_id.split(".", 1)
    if len(parts) == 2 and len(parts[0]) <= 3:
        base_id = parts[1]
        if base_id in BEDROCK_MODEL_METADATA:
            return BEDROCK_MODEL_METADATA[base_id]["max_output_tokens"]

    return 8192


# ---------------------------------------------------------------------------
# boto3 availability check
# ---------------------------------------------------------------------------

def check_boto3_available() -> bool:
    """Check if boto3 is importable. Returns True/False without raising."""
    try:
        import boto3  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _get_tool_call_attr(tc: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute from a tool_call that may be a dict or object."""
    if isinstance(tc, dict):
        return tc.get(attr, default)
    return getattr(tc, attr, default)


def _get_function_field(tc: Any, field: str, default: Any = None) -> Any:
    """Get a field from a tool_call's function (dict or object)."""
    fn = _get_tool_call_attr(tc, "function")
    if fn is None:
        return default
    if isinstance(fn, dict):
        return fn.get(field, default)
    return getattr(fn, field, default)


def _parse_tool_arguments(args: Any) -> dict:
    """Parse tool call arguments — handles both string and dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _convert_image_data_uri(url: str) -> Optional[Dict[str, Any]]:
    """Convert a data URI image to Bedrock image format.

    Parses ``data:<media_type>;base64,<data>`` and returns a Bedrock image
    content block with format and decoded bytes.
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None

    header, sep, data = url.partition(",")
    if not sep or ";base64" not in header:
        return None

    # Extract media type: "data:image/jpeg;base64" → "image/jpeg"
    media_type = header[5:].split(";", 1)[0] or "image/png"

    # Map media type to Bedrock format (jpeg, png, gif, webp)
    fmt_map = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    fmt = fmt_map.get(media_type, "png")

    try:
        decoded = base64.b64decode(data)
    except Exception:
        return None

    return {
        "image": {
            "format": fmt,
            "source": {"bytes": decoded},
        }
    }


def _convert_content_block(part: Any) -> Optional[Dict[str, Any]]:
    """Convert a single OpenAI multi-modal content part to Bedrock format."""
    if not isinstance(part, dict):
        return None

    ptype = part.get("type")

    if ptype == "text":
        text = part.get("text", "")
        if text:
            return {"text": text}
        return None

    if ptype == "image_url":
        image_data = part.get("image_url", {})
        url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data)
        if isinstance(url, str) and url.startswith("data:"):
            return _convert_image_data_uri(url)
        return None

    return None


def _convert_messages_to_bedrock(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict]]:
    """Split OpenAI messages into Bedrock system blocks and conversation messages.

    Returns ``(system_blocks, bedrock_messages)`` where:

    - ``system_blocks`` is a list of ``{"text": "..."}`` dicts extracted from
      system-role messages.
    - ``bedrock_messages`` is a list of Bedrock-format messages with only
      ``user`` and ``assistant`` roles.

    Handles:
    - system messages → separate system parameter
    - user/assistant messages → Bedrock message format with content blocks
    - tool_calls in assistant → toolUse content blocks
    - tool role messages → toolResult content blocks
    - Consecutive same-role merging (Bedrock constraint)
    - Multi-modal content (text + image data URIs)
    """
    system_blocks: List[Dict] = []
    bedrock_messages: List[Dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        # --- System messages → extract into system_blocks ---
        if role == "system":
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            system_blocks.append({"text": text})
                    elif isinstance(part, str) and part:
                        system_blocks.append({"text": part})
            elif isinstance(content, str) and content:
                system_blocks.append({"text": content})
            continue

        # --- Assistant messages ---
        if role == "assistant":
            blocks: List[Dict] = []

            # Text content
            if content:
                if isinstance(content, list):
                    for part in content:
                        block = _convert_content_block(part)
                        if block is not None:
                            blocks.append(block)
                elif isinstance(content, str):
                    blocks.append({"text": content})

            # Tool calls → toolUse blocks
            for tc in m.get("tool_calls", []):
                if tc is None:
                    continue
                tc_id = _get_tool_call_attr(tc, "id", "")
                tc_name = _get_function_field(tc, "name", "")
                tc_args = _get_function_field(tc, "arguments", "{}")
                parsed = _parse_tool_arguments(tc_args)
                blocks.append({
                    "toolUse": {
                        "toolUseId": tc_id,
                        "name": tc_name,
                        "input": parsed,
                    }
                })

            if not blocks:
                blocks = [{"text": "(empty)"}]

            bedrock_messages.append({"role": "assistant", "content": blocks})
            continue

        # --- Tool role messages → toolResult in a user message ---
        if role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            result_text = content if isinstance(content, str) else json.dumps(content)
            if not result_text:
                result_text = "(no output)"

            tool_result_block = {
                "toolResult": {
                    "toolUseId": tool_call_id,
                    "content": [{"text": result_text}],
                }
            }

            bedrock_messages.append({"role": "user", "content": [tool_result_block]})
            continue

        # --- User messages ---
        blocks = []
        if isinstance(content, list):
            for part in content:
                block = _convert_content_block(part)
                if block is not None:
                    blocks.append(block)
        elif isinstance(content, str) and content:
            blocks.append({"text": content})

        if not blocks:
            blocks = [{"text": "(empty message)"}]

        bedrock_messages.append({"role": "user", "content": blocks})

    # --- Merge consecutive same-role messages ---
    merged: List[Dict] = []
    for msg in bedrock_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)

    return system_blocks, merged


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------

# JSON Schema fields that Bedrock does not support
_UNSUPPORTED_SCHEMA_FIELDS = {"additionalProperties", "$schema", "$ref", "$defs"}


def _strip_unsupported_schema_fields(schema: dict) -> dict:
    """Recursively remove JSON Schema fields not supported by Bedrock.

    Removes ``additionalProperties``, ``$schema``, ``$ref``, and ``$defs``
    at every level of the schema tree, including nested ``properties`` values
    and ``items``.

    Returns a new dict — the original is not mutated.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_FIELDS:
            continue

        if key == "properties" and isinstance(value, dict):
            # Recurse into each property's sub-schema
            cleaned[key] = {
                prop_name: _strip_unsupported_schema_fields(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items":
            # items can be a dict (single schema) or list (tuple validation)
            if isinstance(value, dict):
                cleaned[key] = _strip_unsupported_schema_fields(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    _strip_unsupported_schema_fields(item)
                    for item in value
                ]
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value

    return cleaned


def _convert_tools_to_bedrock(tools: List[Dict[str, Any]]) -> List[Dict]:
    """Convert OpenAI function tool definitions to Bedrock toolSpec format.

    Each input tool has the shape::

        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Each output entry has the shape::

        {"toolSpec": {"name": "...", "description": "...", "inputSchema": {"json": <stripped_parameters>}}}

    Unsupported JSON Schema fields are removed from the parameters via
    :func:`_strip_unsupported_schema_fields`.
    """
    bedrock_tools: List[Dict] = []

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        stripped = _strip_unsupported_schema_fields(parameters)

        bedrock_tools.append({
            "toolSpec": {
                "name": name,
                "description": description,
                "inputSchema": {"json": stripped},
            }
        })

    return bedrock_tools


# ---------------------------------------------------------------------------
# Public: build complete Bedrock Converse request kwargs
# ---------------------------------------------------------------------------


def _is_cache_supported_model(model_id: str) -> bool:
    """Check if a resolved Bedrock model ID supports prompt caching.

    Claude models on Bedrock support prompt caching. A model is considered
    cache-supported if its resolved model ID contains ``anthropic.``.
    """
    return "anthropic." in model_id


def _add_cache_points(
    system_blocks: List[Dict],
    bedrock_messages: List[Dict],
) -> None:
    """Add prompt cache markers to system blocks and messages in-place.

    Adds ``{"cachePoint": {"type": "default"}}`` to:
    - The end of the system blocks list (after the last system block)
    - The second-to-last user message's content blocks (if it exists)
    """
    # Add cache point after the last system block
    if system_blocks:
        system_blocks.append({"cachePoint": {"type": "default"}})

    # Find the second-to-last user message and add a cache point
    user_indices = [
        i for i, msg in enumerate(bedrock_messages) if msg.get("role") == "user"
    ]
    if len(user_indices) >= 2:
        second_to_last_idx = user_indices[-2]
        bedrock_messages[second_to_last_idx]["content"].append(
            {"cachePoint": {"type": "default"}}
        )


def build_bedrock_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    reasoning_config: Optional[Dict[str, Any]] = None,
    context_length: Optional[int] = None,
) -> dict:
    """Convert OpenAI-format messages/tools into a Bedrock Converse request body.

    Steps:
    1. Resolve model ID via :func:`get_bedrock_model_id`
    2. Convert messages via :func:`_convert_messages_to_bedrock`
    3. Convert tools via :func:`_convert_tools_to_bedrock` (if provided)
    4. Look up max_tokens from metadata if not specified (fallback 8192)
    5. Build ``inferenceConfig`` with ``maxTokens``
    6. Add prompt cache markers for supported models (Claude on Bedrock)
    7. Return dict with ``modelId``, ``messages``, ``system``, ``inferenceConfig``,
       ``toolConfig``, and optionally ``additionalModelRequestFields``

    Returns a dict ready to be sent to the Bedrock Converse API.
    """
    # 1. Resolve model ID
    model_id = get_bedrock_model_id(model)

    # 2. Convert messages
    system_blocks, bedrock_messages = _convert_messages_to_bedrock(messages)

    # 3. Convert tools if provided
    bedrock_tools = _convert_tools_to_bedrock(tools) if tools else None

    # 4. Resolve max_tokens
    effective_max_tokens = max_tokens or get_bedrock_max_output_tokens(model)

    # Clamp to context window if a lower context_length was specified
    if context_length and effective_max_tokens > context_length:
        effective_max_tokens = max(context_length - 1, 1)

    # 5. Build inferenceConfig
    inference_config: Dict[str, Any] = {
        "maxTokens": effective_max_tokens,
    }

    # 6. Add prompt cache markers for supported models
    if _is_cache_supported_model(model_id):
        _add_cache_points(system_blocks, bedrock_messages)

    # 7. Build the result dict
    kwargs: Dict[str, Any] = {
        "modelId": model_id,
        "messages": bedrock_messages,
        "inferenceConfig": inference_config,
    }

    if system_blocks:
        kwargs["system"] = system_blocks

    if bedrock_tools:
        kwargs["toolConfig"] = {"tools": bedrock_tools}

    # Add reasoning/thinking config if provided and explicitly requested.
    # Only enable Bedrock extended thinking when budget_tokens is explicitly set.
    # OpenRouter-style reasoning configs ({"enabled": True, "effort": "medium"})
    # are NOT translated to Bedrock thinking — they control a different mechanism.
    if reasoning_config and isinstance(reasoning_config, dict):
        budget_tokens = reasoning_config.get("budget_tokens")
        if budget_tokens and isinstance(budget_tokens, int) and budget_tokens > 0:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budgetTokens": budget_tokens,
                }
            }

    return kwargs


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


def _map_stop_reason(bedrock_stop_reason: str) -> str:
    """Map Bedrock stopReason to OpenAI finish_reason.

    Mapping:
    - "end_turn"      → "stop"
    - "tool_use"      → "tool_calls"
    - "max_tokens"    → "length"
    - "stop_sequence" → "stop"
    - anything else   → "stop" (default)
    """
    mapping = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    return mapping.get(bedrock_stop_reason, "stop")


def normalize_bedrock_response(response: dict) -> Tuple[SimpleNamespace, str]:
    """Convert Bedrock Converse response to OpenAI-compatible SimpleNamespace.

    Returns ``(assistant_message, finish_reason)`` matching the shape expected
    by ``AIAgent._build_assistant_message()``.

    The assistant_message SimpleNamespace has:
    - ``.content`` — accumulated text or None
    - ``.tool_calls`` — list of tool call SimpleNamespaces or None
    - ``.reasoning`` — accumulated reasoning text or None
    - ``.reasoning_content`` — None (for compatibility)
    - ``.reasoning_details`` — None (for compatibility)
    - ``.usage`` — SimpleNamespace with token counts
    """
    message = response["output"]["message"]
    content_blocks = message.get("content", [])

    # Accumulate text, tool calls, and reasoning from content blocks
    text_parts: List[str] = []
    tool_calls: List[SimpleNamespace] = []
    reasoning_parts: List[str] = []

    for block in content_blocks:
        # Text block
        if "text" in block:
            text_parts.append(block["text"])

        # Tool use block
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                SimpleNamespace(
                    id=tu["toolUseId"],
                    type="function",
                    function=SimpleNamespace(
                        name=tu["name"],
                        arguments=json.dumps(tu["input"]),
                    ),
                )
            )

        # Reasoning/thinking block
        elif "reasoningContent" in block:
            reasoning_text_block = block["reasoningContent"].get("reasoningText", {})
            text = reasoning_text_block.get("text", "")
            if text:
                reasoning_parts.append(text)

    # Build finish_reason
    finish_reason = _map_stop_reason(response.get("stopReason", "end_turn"))

    # Build usage
    usage_data = response.get("usage", {})
    usage_kwargs = {
        "prompt_tokens": usage_data.get("inputTokens", 0),
        "completion_tokens": usage_data.get("outputTokens", 0),
        "total_tokens": usage_data.get("totalTokens", 0),
    }
    if "cacheReadInputTokenCount" in usage_data:
        usage_kwargs["cache_read_input_tokens"] = usage_data["cacheReadInputTokenCount"]
    if "cacheWriteInputTokenCount" in usage_data:
        usage_kwargs["cache_creation_input_tokens"] = usage_data["cacheWriteInputTokenCount"]

    usage = SimpleNamespace(**usage_kwargs)

    # Assemble the assistant message
    assistant_message = SimpleNamespace(
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
        reasoning="".join(reasoning_parts) if reasoning_parts else None,
        reasoning_content=None,
        reasoning_details=None,
        usage=usage,
    )

    return assistant_message, finish_reason


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


class AuthError(RuntimeError):
    """Raised when AWS credential resolution fails."""
    pass


class BedrockCredentialResolver:
    """Manages AWS credentials with caching and auto-refresh.

    Uses boto3's credential chain internally. Caches the boto3 Session
    and refreshes credentials before expiry (59-minute TTL for temporary creds).
    """

    _CACHE_TTL_SECONDS = 59 * 60  # 59 minutes

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_profile: Optional[str] = None,
        aws_region: Optional[str] = None,
    ):
        """Initialize with optional explicit credentials.

        If no explicit credentials provided, falls back to boto3's default
        credential chain (env vars → AWS profile → IAM instance profile → IRSA).

        Does NOT import boto3 here — that happens lazily in get_credentials().
        """
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._aws_profile = aws_profile
        self._aws_region = aws_region

        # Credential cache
        self._cached_credentials: Optional[Tuple[str, str, Optional[str]]] = None
        self._cached_at: Optional[float] = None

    def get_credentials(self) -> Tuple[str, str, Optional[str]]:
        """Return (access_key, secret_key, session_token_or_None).

        Refreshes credentials if they are within 1 minute of expiry.
        Raises ImportError if boto3 is not installed.
        Raises AuthError if no credentials can be resolved.
        """
        # Lazy-check boto3 availability
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise ImportError(
                "boto3 is required for the Bedrock provider. "
                "Install it with: pip install boto3 or pip install hermes-agent[bedrock]"
            )

        # If explicit credentials were provided, return them directly
        if self._aws_access_key_id and self._aws_secret_access_key:
            return (
                self._aws_access_key_id,
                self._aws_secret_access_key,
                self._aws_session_token,
            )

        # Check cache validity
        if self._cached_credentials is not None and self._cached_at is not None:
            elapsed = time.time() - self._cached_at
            # For temporary credentials (have session_token), use TTL
            # For permanent credentials, also use TTL as a reasonable refresh interval
            if elapsed < self._CACHE_TTL_SECONDS:
                return self._cached_credentials

        # Resolve via boto3 credential chain
        session_kwargs: Dict[str, Any] = {}
        if self._aws_profile:
            session_kwargs["profile_name"] = self._aws_profile

        session = boto3.Session(**session_kwargs)
        credentials = session.get_credentials()

        if credentials is None:
            attempted = ["environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"]
            if self._aws_profile:
                attempted.append(f"AWS profile '{self._aws_profile}'")
            else:
                attempted.append("default AWS profile (~/.aws/credentials)")
            attempted.append("IAM instance profile (EC2/ECS metadata service)")
            attempted.append("EKS IRSA web identity token")
            raise AuthError(
                "No AWS credentials found. Attempted methods:\n"
                + "\n".join(f"  - {m}" for m in attempted)
                + "\n\nPlease configure AWS credentials via environment variables, "
                "AWS profile, or IAM instance role."
            )

        # Resolve credentials (handles RefreshableCredentials too)
        resolved = credentials.get_frozen_credentials()
        access_key = resolved.access_key
        secret_key = resolved.secret_key
        token = resolved.token  # None for permanent credentials

        if not access_key or not secret_key:
            raise AuthError(
                "AWS credentials resolved but access key or secret key is empty. "
                "Please check your AWS credential configuration."
            )

        # Cache the resolved credentials
        self._cached_credentials = (access_key, secret_key, token)
        self._cached_at = time.time()

        return self._cached_credentials

    @property
    def region(self) -> str:
        """Return the resolved AWS region.

        Priority:
        1. Explicit aws_region from constructor
        2. AWS_REGION environment variable
        3. AWS_DEFAULT_REGION environment variable
        4. boto3 session region (if boto3 available)
        5. Default: us-east-1
        """
        if self._aws_region:
            return self._aws_region

        # Try environment variables
        env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if env_region:
            return env_region

        # Try boto3 session region
        try:
            import boto3
            session_kwargs: Dict[str, Any] = {}
            if self._aws_profile:
                session_kwargs["profile_name"] = self._aws_profile
            session = boto3.Session(**session_kwargs)
            if session.region_name:
                return session.region_name
        except ImportError:
            pass

        return "us-east-1"


# ---------------------------------------------------------------------------
# Retry configuration constants
# ---------------------------------------------------------------------------

_RETRY_MAX = 3
_THROTTLE_BASE_DELAY = 1.0
_THROTTLE_MAX_DELAY = 8.0
_SERVICE_ERROR_BASE_DELAY = 5.0


# ---------------------------------------------------------------------------
# SigV4 request signing
# ---------------------------------------------------------------------------


def _sign_request(
    url: str,
    body: bytes,
    credentials: "BedrockCredentialResolver",
    region: str,
) -> Dict[str, str]:
    """Sign an HTTP request using AWS SigV4. Returns signed headers dict.

    Lazy-imports botocore's SigV4Auth, Credentials, and AWSRequest so that
    the module can be imported without botocore installed.
    """
    from botocore.auth import SigV4Auth
    from botocore.credentials import Credentials
    from botocore.awsrequest import AWSRequest

    access_key, secret_key, token = credentials.get_credentials()
    creds = Credentials(access_key, secret_key, token)

    request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(creds, "bedrock", region).add_auth(request)

    return dict(request.headers)


# ---------------------------------------------------------------------------
# Bedrock Converse API client (non-streaming)
# ---------------------------------------------------------------------------


class BedrockAPIError(RuntimeError):
    """Raised when the Bedrock Converse API returns an error response."""

    def __init__(self, status_code: int, error_code: str, message: str, request_id: str = ""):
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        detail = f"[{error_code}] {message}" if error_code else message
        if request_id:
            detail += f" (RequestId: {request_id})"
        super().__init__(f"Bedrock API error {status_code}: {detail}")


def bedrock_converse_create(
    kwargs: dict,
    credentials: "BedrockCredentialResolver",
    region: str,
    endpoint_url: Optional[str] = None,
) -> dict:
    """Send a non-streaming Converse API request via httpx.

    Constructs the endpoint URL, signs the request with SigV4, sends it,
    and handles retries for throttling, transient errors, and credential
    refresh on 403.

    Returns the parsed JSON response on success.
    Raises BedrockAPIError on non-retryable failures.
    """
    import httpx

    # --- Build endpoint URL ---
    model_id = kwargs.get("modelId", "")
    encoded_model_id = url_quote(model_id, safe="")

    base_url = endpoint_url or os.environ.get("AWS_BEDROCK_RUNTIME_ENDPOINT")
    if base_url:
        # Strip trailing slash from custom endpoint
        base_url = base_url.rstrip("/")
    else:
        base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

    url = f"{base_url}/model/{encoded_model_id}/converse"

    # --- Build request body (exclude modelId — it's in the URL) ---
    body_dict = {k: v for k, v in kwargs.items() if k != "modelId"}
    body = json.dumps(body_dict).encode("utf-8")

    # --- Retry loop ---
    credentials_refreshed = False

    for attempt in range(_RETRY_MAX + 1):
        signed_headers = _sign_request(url, body, credentials, region)
        response = httpx.post(url, content=body, headers=signed_headers, timeout=300)

        # Success
        if 200 <= response.status_code < 300:
            return response.json()

        # --- Parse error response ---
        request_id = response.headers.get("x-amzn-requestid", "")
        error_code = ""
        error_message = ""
        try:
            error_body = response.json()
            # Bedrock errors may use "message" or "Message"
            error_code = error_body.get("__type", error_body.get("code", ""))
            # Extract short code from fully-qualified type
            # e.g. "com.amazonaws.bedrock#ThrottlingException" → "ThrottlingException"
            if "#" in error_code:
                error_code = error_code.rsplit("#", 1)[-1]
            error_message = error_body.get("message", error_body.get("Message", ""))
        except Exception:
            error_message = response.text or f"HTTP {response.status_code}"

        # --- 403: try refreshing credentials once ---
        if response.status_code == 403 and not credentials_refreshed:
            credentials_refreshed = True
            # Invalidate cached credentials so next call re-resolves
            credentials._cached_credentials = None
            credentials._cached_at = None
            continue

        # --- Throttling (429) ---
        if response.status_code == 429 or error_code == "ThrottlingException":
            if attempt < _RETRY_MAX:
                delay = min(
                    _THROTTLE_BASE_DELAY * (2 ** attempt),
                    _THROTTLE_MAX_DELAY,
                )
                delay += random.uniform(0, 0.5)
                time.sleep(delay)
                continue

        # --- Service errors (503) ---
        if response.status_code == 503 or error_code in (
            "ModelNotReadyException",
            "ServiceUnavailableException",
        ):
            if attempt < _RETRY_MAX:
                delay = _SERVICE_ERROR_BASE_DELAY * (2 ** attempt)
                delay += random.uniform(0, 0.5)
                time.sleep(delay)
                continue

        # --- Non-retryable or retries exhausted ---
        raise BedrockAPIError(
            status_code=response.status_code,
            error_code=error_code,
            message=error_message,
            request_id=request_id,
        )

    # Should not reach here, but just in case
    raise BedrockAPIError(
        status_code=response.status_code,
        error_code=error_code,
        message=error_message,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Bedrock ConverseStream API client (streaming)
# ---------------------------------------------------------------------------


def _iter_stream_events(response_stream):
    """Yield ``(event_type, payload_dict)`` tuples from an httpx streaming response.

    Uses botocore's ``EventStreamBuffer`` to decode the AWS binary event
    stream framing.  Each yielded event is a ``(str, dict)`` pair where the
    string is the event type (e.g. ``"messageStart"``, ``"contentBlockDelta"``)
    and the dict is the parsed JSON payload.
    """
    import logging
    from botocore.eventstream import EventStreamBuffer

    buf = EventStreamBuffer()

    for chunk in response_stream.iter_bytes():
        buf.add_data(chunk)
        for event_message in buf:
            # event_message is a botocore.eventstream.EventStreamMessage
            # Headers contain :event-type, :content-type, :message-type, etc.
            headers = dict(event_message.headers)
            message_type = headers.get(":message-type", "event")

            if message_type == "exception":
                # Server-side exception delivered over the event stream
                error_code = headers.get(":exception-type", "UnknownException")
                try:
                    error_body = json.loads(event_message.payload)
                    error_msg = error_body.get("message", error_body.get("Message", str(error_body)))
                except Exception:
                    error_msg = event_message.payload.decode("utf-8", errors="replace")
                raise BedrockAPIError(
                    status_code=400,
                    error_code=error_code,
                    message=error_msg,
                )

            if message_type != "event":
                continue

            event_type = headers.get(":event-type", "")
            if not event_type:
                continue

            try:
                payload = json.loads(event_message.payload) if event_message.payload else {}
            except (json.JSONDecodeError, ValueError):
                logging.warning("Bedrock stream: malformed event payload, skipping")
                continue

            yield event_type, payload


def bedrock_converse_stream(
    kwargs: dict,
    credentials: "BedrockCredentialResolver",
    region: str,
    endpoint_url: Optional[str] = None,
    stream_delta_callback: Optional[callable] = None,
    reasoning_callback: Optional[callable] = None,
    tool_gen_callback: Optional[callable] = None,
) -> dict:
    """Send a streaming ConverseStream API request via httpx.

    Fires callbacks for text deltas, reasoning deltas, and tool generation
    events as they arrive.  After all stream events are consumed, assembles
    and returns a complete response dict matching the non-streaming shape
    (i.e. the same structure that :func:`bedrock_converse_create` returns),
    so :func:`normalize_bedrock_response` can process it identically.

    Falls back to :func:`bedrock_converse_create` (non-streaming) if the
    streaming connection is interrupted or times out.

    Parameters
    ----------
    kwargs : dict
        Bedrock Converse request body (from :func:`build_bedrock_kwargs`).
    credentials : BedrockCredentialResolver
        AWS credential resolver.
    region : str
        AWS region for the Bedrock endpoint.
    endpoint_url : str, optional
        Custom Bedrock endpoint URL override.
    stream_delta_callback : callable, optional
        Called with each text delta string as it arrives.
    reasoning_callback : callable, optional
        Called with each reasoning/thinking delta string.
    tool_gen_callback : callable, optional
        Called with the tool name when a toolUse block starts.

    Returns
    -------
    dict
        A response dict matching the non-streaming Converse API shape::

            {
                "output": {"message": {"role": "assistant", "content": [...]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": ..., "outputTokens": ..., "totalTokens": ...},
            }
    """
    import httpx
    import logging

    # --- Build streaming endpoint URL ---
    model_id = kwargs.get("modelId", "")
    encoded_model_id = url_quote(model_id, safe="")

    base_url = endpoint_url or os.environ.get("AWS_BEDROCK_RUNTIME_ENDPOINT")
    if base_url:
        base_url = base_url.rstrip("/")
    else:
        base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

    url = f"{base_url}/model/{encoded_model_id}/converse-stream"

    # --- Build request body (exclude modelId — it's in the URL) ---
    body_dict = {k: v for k, v in kwargs.items() if k != "modelId"}
    body = json.dumps(body_dict).encode("utf-8")

    # --- Sign the request ---
    signed_headers = _sign_request(url, body, credentials, region)

    # --- State for assembling the final response ---
    role = "assistant"
    content_blocks: List[Dict] = []  # Assembled content blocks
    current_block_index: Optional[int] = None
    current_block: Optional[Dict] = None  # Block being accumulated
    # For tool use blocks, we accumulate the JSON input string
    tool_input_buffer: str = ""
    stop_reason: str = "end_turn"
    usage: Dict[str, Any] = {}

    def _finalize_current_block():
        """Finalize the current content block and append to content_blocks."""
        nonlocal current_block, current_block_index, tool_input_buffer

        if current_block is None:
            return

        # If this is a toolUse block, parse the accumulated JSON input
        if "toolUse" in current_block:
            try:
                current_block["toolUse"]["input"] = json.loads(tool_input_buffer) if tool_input_buffer else {}
            except (json.JSONDecodeError, ValueError):
                current_block["toolUse"]["input"] = {}
            tool_input_buffer = ""

        content_blocks.append(current_block)
        current_block = None
        current_block_index = None

    try:
        with httpx.Client(timeout=300) as client:
            with client.stream("POST", url, content=body, headers=signed_headers) as response:
                # Check for HTTP-level errors before parsing the event stream
                if response.status_code >= 400:
                    # Read the full error body
                    error_body_bytes = b""
                    for chunk in response.iter_bytes():
                        error_body_bytes += chunk
                    request_id = response.headers.get("x-amzn-requestid", "")
                    error_code = ""
                    error_message = ""
                    try:
                        error_body = json.loads(error_body_bytes)
                        error_code = error_body.get("__type", error_body.get("code", ""))
                        if "#" in error_code:
                            error_code = error_code.rsplit("#", 1)[-1]
                        error_message = error_body.get("message", error_body.get("Message", ""))
                    except Exception:
                        error_message = error_body_bytes.decode("utf-8", errors="replace") or f"HTTP {response.status_code}"
                    raise BedrockAPIError(
                        status_code=response.status_code,
                        error_code=error_code,
                        message=error_message,
                        request_id=request_id,
                    )

                # --- Parse the event stream ---
                for event_type, payload in _iter_stream_events(response):

                    # --- messageStart ---
                    if event_type == "messageStart":
                        msg_start = payload.get("messageStart", payload)
                        role = msg_start.get("role", "assistant")

                    # --- contentBlockStart ---
                    elif event_type == "contentBlockStart":
                        _finalize_current_block()
                        block_start = payload.get("contentBlockStart", payload)
                        current_block_index = block_start.get("contentBlockIndex", len(content_blocks))
                        start_data = block_start.get("start", {})

                        if "toolUse" in start_data:
                            # Starting a tool use block
                            tu = start_data["toolUse"]
                            current_block = {
                                "toolUse": {
                                    "toolUseId": tu.get("toolUseId", ""),
                                    "name": tu.get("name", ""),
                                    "input": {},  # Will be filled on contentBlockStop
                                }
                            }
                            tool_input_buffer = ""
                            # Fire tool generation callback
                            if tool_gen_callback is not None:
                                try:
                                    tool_gen_callback(tu.get("name", ""))
                                except Exception:
                                    pass
                        else:
                            # Starting a text or reasoning block — we'll know
                            # which kind from the deltas that follow
                            current_block = {}

                    # --- contentBlockDelta ---
                    elif event_type == "contentBlockDelta":
                        block_delta = payload.get("contentBlockDelta", payload)
                        delta = block_delta.get("delta", {})

                        # Auto-initialize current_block if no contentBlockStart was received
                        if current_block is None:
                            current_block = {}
                            current_block_index = block_delta.get("contentBlockIndex", len(content_blocks))

                        # Text delta
                        if "text" in delta:
                            text = delta["text"]
                            # Initialize text block if needed
                            if current_block is not None and "text" not in current_block and "toolUse" not in current_block and "reasoningContent" not in current_block:
                                current_block["text"] = ""
                            if current_block is not None and "text" in current_block:
                                current_block["text"] += text
                            elif current_block is not None and not current_block:
                                # Empty block — this is a text block
                                current_block["text"] = text

                            # Fire stream delta callback
                            if stream_delta_callback is not None:
                                try:
                                    stream_delta_callback(text)
                                except Exception:
                                    pass

                        # Tool use input delta
                        elif "toolUse" in delta:
                            input_chunk = delta["toolUse"].get("input", "")
                            if input_chunk:
                                tool_input_buffer += input_chunk

                        # Reasoning/thinking delta
                        elif "reasoningContent" in delta:
                            reasoning_text = delta["reasoningContent"].get("text", "")
                            if reasoning_text:
                                # Initialize reasoning block if needed
                                if current_block is not None and "reasoningContent" not in current_block and "text" not in current_block and "toolUse" not in current_block:
                                    current_block["reasoningContent"] = {"reasoningText": {"text": ""}}
                                if current_block is not None and "reasoningContent" in current_block:
                                    current_block["reasoningContent"]["reasoningText"]["text"] += reasoning_text
                                elif current_block is not None and not current_block:
                                    current_block["reasoningContent"] = {"reasoningText": {"text": reasoning_text}}

                                # Fire reasoning callback
                                if reasoning_callback is not None:
                                    try:
                                        reasoning_callback(reasoning_text)
                                    except Exception:
                                        pass

                    # --- contentBlockStop ---
                    elif event_type == "contentBlockStop":
                        _finalize_current_block()

                    # --- messageStop ---
                    elif event_type == "messageStop":
                        msg_stop = payload.get("messageStop", payload)
                        stop_reason = msg_stop.get("stopReason", "end_turn")

                    # --- metadata ---
                    elif event_type == "metadata":
                        meta = payload.get("metadata", payload)
                        usage = meta.get("usage", usage)

    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
            httpx.PoolTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError,
            ConnectionError, TimeoutError) as exc:
        # --- Fallback to non-streaming ---
        logging.warning(
            "Bedrock stream interrupted (%s: %s), falling back to non-streaming",
            type(exc).__name__, exc,
        )
        return bedrock_converse_create(
            kwargs=kwargs,
            credentials=credentials,
            region=region,
            endpoint_url=endpoint_url,
        )

    # --- Finalize any remaining open block ---
    _finalize_current_block()

    # --- Assemble the complete response dict ---
    # Compute totalTokens if not provided
    if "totalTokens" not in usage:
        usage["totalTokens"] = usage.get("inputTokens", 0) + usage.get("outputTokens", 0)

    return {
        "output": {
            "message": {
                "role": role,
                "content": content_blocks,
            }
        },
        "stopReason": stop_reason,
        "usage": usage,
    }
