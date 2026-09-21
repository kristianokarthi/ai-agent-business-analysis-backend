import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "nvidia/nemotron-3-super-120b-a12b",
    )
    openrouter_chat_model: str = os.getenv(
        "OPENROUTER_CHAT_MODEL",
        "nvidia/nemotron-3-super-120b-a12b:free",
    )
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )
    gemini_embedding_dimensions: int = int(
        os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768")
    )
    rag_min_similarity: float = float(
        os.getenv("RAG_MIN_SIMILARITY", "0.35")
    )
    research_tool_test_key: str = os.getenv(
        "RESEARCH_TOOL_TEST_KEY",
        "",
    )
    frontend_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            (
                "http://localhost:3000,"
                "https://ai-business-frontend.vercel.app"
            ),
        ).split(",")
        if origin.strip()
    ]


settings = Settings()
