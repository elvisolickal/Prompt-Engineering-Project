"""
Central configuration — loads from .env file automatically.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Available Models ──────────────────────────────────────────
OPENAI_MODELS: dict = {
    "GPT-4o": "gpt-4o",
    "GPT-4o Mini": "gpt-4o-mini",
}

GEMINI_MODELS: dict = {
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
}

ALL_MODELS: dict = {**OPENAI_MODELS, **GEMINI_MODELS}

# ── Genetic Algorithm Defaults ─────────────────────────────────
GA_POPULATION_SIZE: int = 12
GA_MAX_GENERATIONS: int = 4
GA_ELITE_FRACTION: float = 0.33
# Cheap, fast model used for mutation/crossover operations
GA_MUTATION_MODEL: str = "gpt-4o-mini"
GA_MAX_TOKENS_PER_CALL: int = 1200

# ── Similarity Scoring Weights ────────────────────────────────
SCORE_WEIGHT_FEATURE_DIST: float = 0.35
SCORE_WEIGHT_BURROWS_DELTA: float = 0.25
SCORE_WEIGHT_EMBEDDING: float = 0.20
SCORE_WEIGHT_TFIDF: float = 0.10
SCORE_WEIGHT_READABILITY: float = 0.10

# ── Paths ─────────────────────────────────────────────────────
DB_PATH: str = "results/experiments.db"
SAMPLES_DIR: str = "data/samples"
RESULTS_DIR: str = "results"
