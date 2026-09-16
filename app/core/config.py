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
