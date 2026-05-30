"""
Similarity Scorer — compare generated text against an author's StyleProfile.

Returns a composite Style Match Score (0–100) combining:
  - Feature vector distance   35 %
  - Burrows' Delta            25 %
  - Sentence-embedding cosine 20 %
  - TF-IDF cosine (n-grams)  10 %
  - Readability alignment     10 %
"""
import re
import math
from collections import Counter
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .style_analyzer import StyleProfile, StyleAnalyzer, FUNCTION_WORDS
from .ingestion import CorpusIngester, Document

import config


class SimilarityScorer:
    """Score how closely a generated text matches an author's StyleProfile."""

    def __init__(self):
        self._ingester = CorpusIngester()
        self._analyzer = StyleAnalyzer()
        self._embed_model = None          # lazy-loaded
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
        )
        self._reference_texts: List[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, reference_profile: StyleProfile, corpus_text: str) -> None:
        """Call once with the author's corpus before scoring any generated text."""
        self._reference_profile = reference_profile
        self._reference_texts = self._split_sentences(corpus_text)
        # Fit TF-IDF on reference corpus
        if self._reference_texts:
            self._tfidf.fit(self._reference_texts)

    def score(self, generated_text: str) -> Dict[str, float]:
        """
        Score generated_text against the fitted reference profile.
        Returns a dict with individual dimension scores and the composite.
        """
        if not generated_text.strip():
            return {"composite": 0.0}

        # Build a mini-profile for the generated text
        doc = self._ingester.load_from_text(generated_text, "generated")
        if doc is None:
            doc = Document(
                filename="generated",
                content=generated_text,
                paragraphs=[generated_text],
                sentences=self._split_sentences(generated_text),
                word_count=len(generated_text.split()),
                source_type="generated",
            )
        gen_profile = self._analyzer.analyze([doc], author_name="generated")

        scores = {}

        # 1 · Feature vector distance → similarity
        scores["feature_dist"] = self._feature_similarity(
            self._reference_profile, gen_profile
        )

        # 2 · Burrows' Delta (function word frequencies)
        scores["burrows_delta"] = self._burrows_similarity(
            self._reference_profile, gen_profile
        )

        # 3 · Sentence-embedding cosine similarity
        scores["embedding"] = self._embedding_similarity(generated_text)

        # 4 · TF-IDF cosine (surface n-gram overlap)
        scores["tfidf"] = self._tfidf_similarity(generated_text)

        # 5 · Readability alignment
        scores["readability"] = self._readability_similarity(
            self._reference_profile, gen_profile
        )

        # Composite (weighted average → 0–100)
        w = config
        composite = (
            scores["feature_dist"]  * w.SCORE_WEIGHT_FEATURE_DIST
            + scores["burrows_delta"] * w.SCORE_WEIGHT_BURROWS_DELTA
            + scores["embedding"]     * w.SCORE_WEIGHT_EMBEDDING
            + scores["tfidf"]         * w.SCORE_WEIGHT_TFIDF
            + scores["readability"]   * w.SCORE_WEIGHT_READABILITY
        ) * 100

        scores["composite"] = round(min(max(composite, 0), 100), 2)
        return scores

    # ── Dimension scorers (each returns 0–1) ──────────────────────────────────

    def _feature_similarity(self, ref: StyleProfile, gen: StyleProfile) -> float:
        ref_vec = np.array(ref.get_numeric_vector(), dtype=float)
        gen_vec = np.array(gen.get_numeric_vector(), dtype=float)

        # Normalize each dimension by the reference value to get relative distances
        epsilon = 1e-8
        diffs = np.abs(ref_vec - gen_vec) / (np.abs(ref_vec) + epsilon)
        # Mean relative error → clamp → invert
        mean_err = np.clip(np.mean(diffs), 0, 1)
        return 1.0 - mean_err

    def _burrows_similarity(self, ref: StyleProfile, gen: StyleProfile) -> float:
        """Simplified Burrows' Delta — lower delta = more similar."""
        ref_fw = ref.function_word_freqs
        gen_fw = gen.function_word_freqs

        if not ref_fw or not gen_fw:
            return 0.5

        # Standard deviation of each function word across reference
        ref_vals = np.array([ref_fw.get(fw, 0) for fw in FUNCTION_WORDS], dtype=float)
        gen_vals = np.array([gen_fw.get(fw, 0) for fw in FUNCTION_WORDS], dtype=float)

        std = np.std(ref_vals) + 1e-8
        delta = np.mean(np.abs(ref_vals - gen_vals) / std)
        # Convert delta to similarity: delta=0 → 1.0, delta=2 → 0.0
        return max(0.0, 1.0 - delta / 2.0)

    def _embedding_similarity(self, generated_text: str) -> float:
        """Semantic similarity via sentence-transformers (lazy-loaded)."""
        try:
            if self._embed_model is None:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

            ref_sample = " ".join(self._reference_texts[:50])[:3000]
            gen_sample = generated_text[:3000]

            embeddings = self._embed_model.encode(
                [ref_sample, gen_sample], convert_to_numpy=True
            )
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.5  # neutral fallback

    def _tfidf_similarity(self, generated_text: str) -> float:
        """TF-IDF cosine similarity between generated text and reference corpus."""
        try:
            gen_sentences = self._split_sentences(generated_text)
            if not gen_sentences or not self._reference_texts:
                return 0.5

            gen_vec = self._tfidf.transform([" ".join(gen_sentences[:20])])
            ref_vec = self._tfidf.transform([" ".join(self._reference_texts[:50])])
            sim = cosine_similarity(gen_vec, ref_vec)[0][0]
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.5

    def _readability_similarity(self, ref: StyleProfile, gen: StyleProfile) -> float:
        """Closeness of Flesch reading-ease and Gunning Fog scores."""
        flesch_diff = abs(ref.flesch_reading_ease - gen.flesch_reading_ease) / 100
        fog_diff = abs(ref.gunning_fog - gen.gunning_fog) / 20
        mean_diff = (flesch_diff + fog_diff) / 2
        return max(0.0, 1.0 - mean_diff)

    # ── Helper ────────────────────────────────────────────────────────────────
    def _split_sentences(self, text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            return re.split(r"(?<=[.!?])\s+", text)

    def get_weak_dimensions(self, scores: Dict[str, float]) -> List[str]:
        """Return names of the lowest-scoring dimensions (for GA mutation hints)."""
        dim_map = {
            "feature_dist": "lexical and syntactic feature match",
            "burrows_delta": "function word distribution (Burrows' Delta)",
            "embedding": "semantic and stylistic similarity",
            "tfidf": "vocabulary and n-gram overlap",
            "readability": "readability level alignment",
        }
        ranked = sorted(
            [(k, v) for k, v in scores.items() if k != "composite"],
            key=lambda x: x[1],
        )
        return [dim_map.get(k, k) for k, _ in ranked[:2]]
