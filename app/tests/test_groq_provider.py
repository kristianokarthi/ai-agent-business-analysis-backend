from app.core.config import settings
from app.llm.groq_provider import GroqProvider


def test_groq_provider_initializes(monkeypatch):
    monkeypatch.setattr(
        settings,
        "groq_api_key",
        "test-api-key",
    )

    provider = GroqProvider()

    assert provider.model == settings.groq_model
