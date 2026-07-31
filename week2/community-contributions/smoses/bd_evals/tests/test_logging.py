"""Tests for secret redaction in logging."""

from rag_evals.logging import is_secret_key, redact, setup_logging


def test_is_secret_key_matches() -> None:
    assert is_secret_key("api_key")
    assert is_secret_key("API_KEY")
    assert is_secret_key("api_key_env")
    assert is_secret_key("token")
    assert is_secret_key("secret")
    assert is_secret_key("password")
    assert is_secret_key("credential")
    assert is_secret_key("my_secret_value")


def test_is_secret_key_does_not_match() -> None:
    assert not is_secret_key("base_url_env")
    assert not is_secret_key("description")
    assert not is_secret_key("question")
    assert not is_secret_key("mode")
    assert not is_secret_key("id")


def test_redact_flat_dict() -> None:
    data = {"api_key": "sk-12345", "base_url": "https://example.com"}
    result = redact(data)
    assert result["api_key"] == "[REDACTED]"
    assert result["base_url"] == "https://example.com"


def test_redact_nested_dict() -> None:
    data = {
        "api": {
            "api_key": "sk-12345",
            "api_key_env": "BD_API_KEY",
            "base_url_env": "BD_API_BASE_URL",
        },
        "questions": [{"id": "q001"}],
    }
    result = redact(data)
    assert result["api"]["api_key"] == "[REDACTED]"
    assert result["api"]["api_key_env"] == "[REDACTED]"
    assert result["api"]["base_url_env"] == "BD_API_BASE_URL"
    assert result["questions"][0]["id"] == "q001"


def test_redact_list_of_dicts() -> None:
    data = [
        {"api_key": "secret1", "name": "a"},
        {"api_key": "secret2", "name": "b"},
    ]
    result = redact(data)
    assert result[0]["api_key"] == "[REDACTED]"
    assert result[0]["name"] == "a"
    assert result[1]["api_key"] == "[REDACTED]"
    assert result[1]["name"] == "b"


def test_redact_non_dict_passthrough() -> None:
    assert redact("hello") == "hello"
    assert redact(42) == 42
    assert redact(None) is None
    assert redact(True) is True


def test_redact_does_not_mutate_input() -> None:
    data = {"api_key": "sk-12345", "name": "test"}
    result = redact(data)
    assert data["api_key"] == "sk-12345"
    assert result["api_key"] == "[REDACTED]"


def test_redact_empty() -> None:
    assert redact({}) == {}
    assert redact([]) == []


def test_setup_logging_returns_logger() -> None:
    logger = setup_logging("DEBUG")
    assert logger.name == "rag_evals"
    assert logger.level == 10  # DEBUG
