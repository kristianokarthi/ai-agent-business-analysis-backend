from app.core.config import settings
from app.llm.groq_provider import GroqProvider

def test_groq_provider_initializes(monkeypatch):
    fake_client = object()

    monkeypatch.setattr(
        settings,
        "groq_api_key",
        "test-api-key",
    )
    monkeypatch.setattr(
        "app.llm.groq_provider.AsyncGroq",
        lambda api_key: fake_client,
    )

    provider = GroqProvider()
    assert provider.client is fake_client
    assert provider.model == settings.groq_model
