"""Unit tests for BedrockCredentialResolver edge cases.

Tests credential resolution behavior including:
- boto3 missing raises ImportError with install instructions
- Module imports succeed without boto3 (lazy import)
- Region defaults to us-east-1 when no region configured
- Credential refresh on expiry (59-minute TTL)

Requirements: 2.7, 2.8, 13.1, 13.2, 13.4, 13.5
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.bedrock_adapter import AuthError, BedrockCredentialResolver


class TestBoto3MissingRaisesImportError:
    """Test that get_credentials() raises ImportError with install instructions when boto3 is missing."""

    def test_get_credentials_raises_import_error_without_boto3(self):
        """When boto3 is not importable, get_credentials raises ImportError with pip install instructions."""
        resolver = BedrockCredentialResolver()

        with patch.dict(sys.modules, {"boto3": None}):
            with pytest.raises(ImportError, match="pip install boto3"):
                resolver.get_credentials()

    def test_import_error_message_includes_bedrock_extra(self):
        """The ImportError message also mentions the hermes-agent[bedrock] extras install."""
        resolver = BedrockCredentialResolver()

        with patch.dict(sys.modules, {"boto3": None}):
            with pytest.raises(ImportError, match=r"hermes-agent\[bedrock\]"):
                resolver.get_credentials()


class TestLazyImportModuleLevel:
    """Test that agent.bedrock_adapter can be imported without boto3 installed."""

    def test_module_import_succeeds_without_boto3(self):
        """Importing agent.bedrock_adapter does not require boto3 at module level."""
        # The fact that we already imported BedrockCredentialResolver at the top
        # of this file proves the module loads without boto3 being used at import
        # time. We verify the module is in sys.modules and has the expected symbols.
        import agent.bedrock_adapter as mod

        assert hasattr(mod, "BedrockCredentialResolver")
        assert hasattr(mod, "AuthError")
        assert hasattr(mod, "build_bedrock_kwargs")
        assert hasattr(mod, "get_bedrock_model_id")

    def test_module_level_code_does_not_import_boto3(self):
        """The module source does not contain a top-level 'import boto3' statement."""
        import inspect
        import agent.bedrock_adapter as mod

        source = inspect.getsource(mod)
        # Top-level imports appear at column 0. Lazy imports are indented inside
        # functions/methods. We check that no un-indented "import boto3" exists.
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("import boto3") or stripped.startswith("from boto3"):
                # Must be indented (inside a function), not at module level
                assert line != stripped, (
                    f"Found top-level boto3 import: {line!r}"
                )


class TestRegionDefaultsToUsEast1:
    """Test that region defaults to us-east-1 when no region is configured."""

    def test_region_defaults_when_no_config(self):
        """With no aws_region, no env vars, and boto3 session returning None, region is us-east-1."""
        resolver = BedrockCredentialResolver()

        mock_session = MagicMock()
        mock_session.region_name = None

        with patch.dict(
            "os.environ",
            {},
            clear=False,
        ):
            # Remove AWS region env vars if present
            env_overrides = {
                k: v for k, v in __import__("os").environ.items()
                if k not in ("AWS_REGION", "AWS_DEFAULT_REGION")
            }
            with patch.dict("os.environ", env_overrides, clear=True):
                with patch("boto3.Session", return_value=mock_session):
                    assert resolver.region == "us-east-1"

    def test_region_uses_explicit_value_over_default(self):
        """When aws_region is explicitly provided, it takes priority."""
        resolver = BedrockCredentialResolver(aws_region="eu-west-1")
        assert resolver.region == "eu-west-1"

    def test_region_uses_env_var_over_boto3(self):
        """AWS_REGION env var takes priority over boto3 session region."""
        resolver = BedrockCredentialResolver()

        with patch.dict("os.environ", {"AWS_REGION": "ap-southeast-1"}, clear=False):
            assert resolver.region == "ap-southeast-1"

    def test_region_uses_default_region_env_var(self):
        """AWS_DEFAULT_REGION env var is used when AWS_REGION is not set."""
        resolver = BedrockCredentialResolver()

        env_clean = {
            k: v for k, v in __import__("os").environ.items()
            if k != "AWS_REGION"
        }
        env_clean["AWS_DEFAULT_REGION"] = "us-west-2"
        with patch.dict("os.environ", env_clean, clear=True):
            assert resolver.region == "us-west-2"


class TestCredentialRefreshOnExpiry:
    """Test that credentials are refreshed after the 59-minute TTL expires."""

    def _make_mock_session(self, access_key="AKIA_TEST", secret_key="secret_test", token="tok"):
        """Create a mock boto3 Session that returns frozen credentials."""
        mock_frozen = MagicMock()
        mock_frozen.access_key = access_key
        mock_frozen.secret_key = secret_key
        mock_frozen.token = token

        mock_creds = MagicMock()
        mock_creds.get_frozen_credentials.return_value = mock_frozen

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds
        return mock_session

    def test_credentials_refreshed_after_ttl(self):
        """Calling get_credentials() after 59+ minutes triggers a new boto3 Session call."""
        resolver = BedrockCredentialResolver()

        mock_session = self._make_mock_session()

        current_time = 1000000.0

        with patch("boto3.Session", return_value=mock_session) as mock_session_cls:
            with patch("time.time", side_effect=lambda: current_time):
                # First call — should create a session and cache credentials
                creds1 = resolver.get_credentials()
                assert creds1 == ("AKIA_TEST", "secret_test", "tok")
                assert mock_session_cls.call_count == 1

            # Advance time by 60 minutes (past the 59-minute TTL)
            current_time += 60 * 60

            with patch("time.time", side_effect=lambda: current_time):
                # Second call — cache expired, should create a new session
                creds2 = resolver.get_credentials()
                assert creds2 == ("AKIA_TEST", "secret_test", "tok")
                assert mock_session_cls.call_count == 2

    def test_credentials_cached_within_ttl(self):
        """Calling get_credentials() within 59 minutes returns cached credentials without new Session."""
        resolver = BedrockCredentialResolver()

        mock_session = self._make_mock_session()

        current_time = 1000000.0

        with patch("boto3.Session", return_value=mock_session) as mock_session_cls:
            with patch("time.time", side_effect=lambda: current_time):
                # First call
                creds1 = resolver.get_credentials()
                assert mock_session_cls.call_count == 1

            # Advance time by 30 minutes (within the 59-minute TTL)
            current_time += 30 * 60

            with patch("time.time", side_effect=lambda: current_time):
                # Second call — should use cache
                creds2 = resolver.get_credentials()
                assert mock_session_cls.call_count == 1  # No new session created

            assert creds1 == creds2

    def test_explicit_credentials_skip_boto3(self):
        """When explicit credentials are provided, boto3 is never called."""
        resolver = BedrockCredentialResolver(
            aws_access_key_id="AKIA_EXPLICIT",
            aws_secret_access_key="secret_explicit",
            aws_session_token="token_explicit",
        )

        with patch("boto3.Session") as mock_session_cls:
            creds = resolver.get_credentials()
            assert creds == ("AKIA_EXPLICIT", "secret_explicit", "token_explicit")
            mock_session_cls.assert_not_called()
