import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )


settings = Settings()