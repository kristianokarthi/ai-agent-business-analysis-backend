from app.llm.groq_provider import GroqProvider


def test_groq_provider_initializes():
    provider = GroqProvider()

    assert provider.model == "openai/gpt-oss-20b"