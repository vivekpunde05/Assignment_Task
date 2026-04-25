"""Configuration module for LLM provider selection and embedding setup."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Bot Personas
BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, "
            "Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical "
            "of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in "
            "finance jargon and view everything through the lens of ROI."
        ),
    },
}

# Embedding model (local, free)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Similarity threshold for Phase 1 routing
SIMILARITY_THRESHOLD = 0.55
