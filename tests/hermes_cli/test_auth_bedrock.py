"""Tests for Bedrock provider registration and alias resolution in hermes_cli.auth."""

import os
import pytest
from unittest.mock import patch


class TestBedrockProviderRegistry:
    """PROVIDER_REGISTRY contains a correct 'bedrock' entry."""

    def test_bedrock_in_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "bedrock" in PROVIDER_REGISTRY

    def test_bedrock_id(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["bedrock"].id == "bedrock"

    def test_bedrock_auth_type(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["bedrock"].auth_type == "api_key"

    def test_bedrock_api_key_env_vars(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["bedrock"].api_key_env_vars == ("AWS_ACCESS_KEY_ID",)

    def test_bedrock_base_url_env_var(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["bedrock"].base_url_env_var == "AWS_BEDROCK_RUNTIME_ENDPOINT"


class TestBedrockProviderAliases:
    """resolve_provider() maps 'bedrock', 'aws-bedrock', and 'aws' to 'bedrock'."""

    def test_resolve_bedrock(self, monkeypatch):
        from hermes_cli.auth import resolve_provider
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        assert resolve_provider("bedrock") == "bedrock"

    def test_resolve_aws_bedrock(self, monkeypatch):
        from hermes_cli.auth import resolve_provider
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        assert resolve_provider("aws-bedrock") == "bedrock"

    def test_resolve_aws(self, monkeypatch):
        from hermes_cli.auth import resolve_provider
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        assert resolve_provider("aws") == "bedrock"
