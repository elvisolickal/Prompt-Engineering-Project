"""
Results Store — SQLite persistence for all experiments.
Uses SQLAlchemy Core (no ORM) for simplicity and Streamlit compatibility.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, Text, DateTime,
    insert, select, desc, text,
)


class ResultsStore:
    """Persist generation experiments and retrieve leaderboard data."""

    def __init__(self, db_path: str = "results/experiments.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.meta = MetaData()
        self._define_tables()
        self.meta.create_all(self.engine)

    # ── Schema ────────────────────────────────────────────────────────────────

    def _define_tables(self):
        self.experiments = Table(
            "experiments", self.meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String(64)),
            Column("author_name", String(256)),
            Column("task", Text),
            Column("model", String(128)),
            Column("strategy", String(128)),
            Column("generation", Integer, default=0),
            Column("system_prompt", Text),
            Column("output", Text),
            Column("score", Float),
            Column("score_feature_dist", Float, nullable=True),
            Column("score_burrows", Float, nullable=True),
            Column("score_embedding", Float, nullable=True),
            Column("score_tfidf", Float, nullable=True),
            Column("score_readability", Float, nullable=True),
            Column("created_at", DateTime, default=datetime.utcnow),
            extend_existing=True,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_result(self, result: Dict, run_id: str, author_name: str, task: str, generation: int = 0):
        dim = result.get("dim_scores", {})
        row = {
            "run_id": run_id,
            "author_name": author_name,
            "task": task[:500],
            "model": result.get("model", ""),
            "strategy": result.get("strategy", ""),
            "generation": generation,
            "system_prompt": result.get("system_prompt", "")[:3000],
            "output": result.get("output", "")[:4000],
            "score": result.get("score", 0.0),
            "score_feature_dist": dim.get("feature_dist"),
            "score_burrows": dim.get("burrows_delta"),
            "score_embedding": dim.get("embedding"),
            "score_tfidf": dim.get("tfidf"),
            "score_readability": dim.get("readability"),
            "created_at": datetime.utcnow(),
        }
        with self.engine.connect() as conn:
            conn.execute(insert(self.experiments), row)
            conn.commit()

    def save_results_batch(self, results: List[Dict], run_id: str, author_name: str, task: str, generation: int = 0):
        for r in results:
            self.save_result(r, run_id, author_name, task, generation)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_leaderboard(self, author_name: Optional[str] = None, limit: int = 50) -> List[Dict]:
        with self.engine.connect() as conn:
            q = select(self.experiments).order_by(desc(self.experiments.c.score)).limit(limit)
            if author_name:
                q = q.where(self.experiments.c.author_name == author_name)
            rows = conn.execute(q).mappings().all()
        return [dict(r) for r in rows]

    def get_best_prompts(self, author_name: str, top_n: int = 5) -> List[Dict]:
        """Return the top-N unique system prompts by score for an author."""
        leaderboard = self.get_leaderboard(author_name, limit=200)
        seen = set()
        best = []
        for row in leaderboard:
            key = row["system_prompt"][:200]
            if key not in seen:
                seen.add(key)
                best.append(row)
            if len(best) >= top_n:
                break
        return best

    def get_run_history(self, run_id: str) -> List[Dict]:
        with self.engine.connect() as conn:
            q = (
                select(self.experiments)
                .where(self.experiments.c.run_id == run_id)
                .order_by(self.experiments.c.generation, desc(self.experiments.c.score))
            )
            rows = conn.execute(q).mappings().all()
        return [dict(r) for r in rows]

    def get_all_authors(self) -> List[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT author_name FROM experiments ORDER BY author_name")
            ).fetchall()
        return [r[0] for r in rows]

    def get_model_comparison(self, author_name: str) -> List[Dict]:
        """Average score per model for a given author."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT model, AVG(score) as avg_score, MAX(score) as max_score, COUNT(*) as runs "
                    "FROM experiments WHERE author_name = :name GROUP BY model ORDER BY avg_score DESC"
                ),
                {"name": author_name},
            ).mappings().all()
        return [dict(r) for r in rows]
