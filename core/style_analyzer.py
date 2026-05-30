"""
Style Analyzer — extract 25+ measurable writing features from a corpus.
Produces a StyleProfile (numerical fingerprint) that drives all scoring
and prompt generation downstream.
"""
import re
import math
import statistics
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

import nltk
import numpy as np

# Bootstrap required NLTK data
for _res in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(_res, quiet=True)
    except Exception:
        pass

# ── Vocabulary lists ───────────────────────────────────────────────────────────
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "consequently", "subsequently", "additionally", "nonetheless", "thus",
    "hence", "meanwhile", "although", "whereas", "conversely", "similarly",
    "in contrast", "as a result", "for instance", "in addition", "in conclusion",
    "to summarize", "on the other hand", "in other words", "in fact", "indeed",
    "for example", "specifically", "notably", "importantly", "ultimately",
}

HEDGE_WORDS = {
    "perhaps", "maybe", "possibly", "probably", "likely", "seems", "appears",
    "might", "could", "would", "tend", "tends", "generally", "typically",
    "often", "sometimes", "usually", "somewhat", "rather", "fairly", "quite",
    "relatively", "approximately", "around", "about", "suggest", "suggests",
    "indicate", "indicates",
}

FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "shall",
    "that", "this", "these", "those", "it", "its", "which", "who", "what",
    "when", "where", "how", "why", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "every", "any", "all", "some",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "his", "her", "their", "if", "because", "although", "while", "since",
]

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}

COMMON_WORDS: set = set()  # populated lazily from NLTK stopwords


def _get_common_words() -> set:
    global COMMON_WORDS
    if not COMMON_WORDS:
        from nltk.corpus import stopwords
        COMMON_WORDS = set(stopwords.words("english"))
    return COMMON_WORDS


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class StyleProfile:
    author_name: str = "Unknown Author"

    # Lexical
    avg_word_length: float = 0.0
    type_token_ratio: float = 0.0
    hapax_legomena_ratio: float = 0.0
    rare_word_ratio: float = 0.0

    # Syntactic
    avg_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    long_sentence_ratio: float = 0.0   # sentences > 25 words
    short_sentence_ratio: float = 0.0  # sentences < 8 words

    # Punctuation (per 100 words)
    comma_per_100: float = 0.0
    semicolon_per_100: float = 0.0
    em_dash_per_100: float = 0.0
    exclamation_per_100: float = 0.0
    question_per_100: float = 0.0
    parenthetical_ratio: float = 0.0
    ellipsis_per_100: float = 0.0

    # Tone
    sentiment_polarity: float = 0.0
    sentiment_subjectivity: float = 0.0
    formality_score: float = 0.0
    hedge_word_ratio: float = 0.0
    first_person_ratio: float = 0.0

    # Structure
    avg_paragraph_length: float = 0.0
    transition_word_ratio: float = 0.0
    rhetorical_question_ratio: float = 0.0

    # Readability
    flesch_reading_ease: float = 0.0
    gunning_fog: float = 0.0

    # Function-word frequencies (Burrows' Delta)
    function_word_freqs: Dict[str, float] = field(default_factory=dict)

    # Few-shot examples for prompts (short passages ≤150 words each)
    sample_passages: List[str] = field(default_factory=list)

    # Human-readable description generated later by LLM or heuristic
    style_description: str = ""

    # Corpus stats
    total_words: int = 0
    total_sentences: int = 0
    total_documents: int = 0

    # ── Helpers ───────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_numeric_vector(self) -> List[float]:
        return [
            self.avg_word_length, self.type_token_ratio, self.hapax_legomena_ratio,
            self.rare_word_ratio, self.avg_sentence_length, self.sentence_length_variance,
            self.long_sentence_ratio, self.short_sentence_ratio,
            self.comma_per_100, self.semicolon_per_100, self.em_dash_per_100,
            self.exclamation_per_100, self.question_per_100, self.parenthetical_ratio,
            self.ellipsis_per_100, self.sentiment_polarity, self.sentiment_subjectivity,
            self.formality_score, self.hedge_word_ratio, self.first_person_ratio,
            self.avg_paragraph_length, self.transition_word_ratio,
            self.rhetorical_question_ratio, self.flesch_reading_ease, self.gunning_fog,
        ]

    def get_feature_names(self) -> List[str]:
        return [
            "avg_word_length", "type_token_ratio", "hapax_legomena_ratio",
            "rare_word_ratio", "avg_sentence_length", "sentence_length_variance",
            "long_sentence_ratio", "short_sentence_ratio",
            "comma_per_100", "semicolon_per_100", "em_dash_per_100",
            "exclamation_per_100", "question_per_100", "parenthetical_ratio",
            "ellipsis_per_100", "sentiment_polarity", "sentiment_subjectivity",
            "formality_score", "hedge_word_ratio", "first_person_ratio",
            "avg_paragraph_length", "transition_word_ratio",
            "rhetorical_question_ratio", "flesch_reading_ease", "gunning_fog",
        ]


# ── Analyzer ───────────────────────────────────────────────────────────────────
class StyleAnalyzer:
    """Extract a StyleProfile from a list of Document objects."""

    def analyze(self, documents, author_name: str = "Author") -> StyleProfile:
        all_text = "\n\n".join(d.content for d in documents)
        all_sentences: List[str] = []
        all_paragraphs: List[str] = []
        for doc in documents:
            all_sentences.extend(doc.sentences)
            all_paragraphs.extend(doc.paragraphs)

        words_raw = re.findall(r"\b[a-zA-Z]+\b", all_text)
        words_lower = [w.lower() for w in words_raw]
        total_words = len(words_lower)

        p = StyleProfile(author_name=author_name)
        p.total_documents = len(documents)
        p.total_sentences = len(all_sentences)
        p.total_words = total_words

        if total_words == 0:
            return p

        # ── Lexical ──────────────────────────────────────────────────────────
        p.avg_word_length = sum(len(w) for w in words_lower) / total_words
        word_counts = Counter(words_lower)
        unique_words = len(word_counts)
        p.type_token_ratio = unique_words / total_words if total_words else 0
        hapax = sum(1 for c in word_counts.values() if c == 1)
        p.hapax_legomena_ratio = hapax / unique_words if unique_words else 0
        common = _get_common_words()
        rare = sum(1 for w in words_lower if w not in common and len(w) > 6)
        p.rare_word_ratio = rare / total_words

        # ── Syntactic ────────────────────────────────────────────────────────
        sent_lengths = [len(s.split()) for s in all_sentences]
        if sent_lengths:
            p.avg_sentence_length = statistics.mean(sent_lengths)
            p.sentence_length_variance = statistics.variance(sent_lengths) if len(sent_lengths) > 1 else 0.0
            p.long_sentence_ratio = sum(1 for l in sent_lengths if l > 25) / len(sent_lengths)
            p.short_sentence_ratio = sum(1 for l in sent_lengths if l < 8) / len(sent_lengths)

        # ── Punctuation ──────────────────────────────────────────────────────
        per100 = 100 / total_words if total_words else 0
        p.comma_per_100 = all_text.count(",") * per100
        p.semicolon_per_100 = all_text.count(";") * per100
        p.em_dash_per_100 = (all_text.count("—") + all_text.count(" -- ")) * per100
        p.exclamation_per_100 = all_text.count("!") * per100
        p.question_per_100 = all_text.count("?") * per100
        p.ellipsis_per_100 = (all_text.count("...") + all_text.count("…")) * per100
        paren_opens = all_text.count("(")
        p.parenthetical_ratio = paren_opens / len(all_sentences) if all_sentences else 0

        # ── Tone ─────────────────────────────────────────────────────────────
        try:
            from textblob import TextBlob
            blob = TextBlob(all_text[:10000])  # cap for speed
            p.sentiment_polarity = blob.sentiment.polarity
            p.sentiment_subjectivity = blob.sentiment.subjectivity
        except Exception:
            pass

        p.hedge_word_ratio = sum(1 for w in words_lower if w in HEDGE_WORDS) / total_words
        p.first_person_ratio = sum(1 for w in words_lower if w in FIRST_PERSON) / total_words

        # Formality: ratio of nouns+adjectives to pronouns+adverbs (Heylighen & Dewaele)
        try:
            sample_words = words_raw[:2000]
            tagged = nltk.pos_tag(sample_words)
            formal_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
            informal_pos = {"PRP", "PRP$", "RB", "RBR", "RBS"}
            formal_count = sum(1 for _, t in tagged if t in formal_pos)
            informal_count = sum(1 for _, t in tagged if t in informal_pos)
            denom = formal_count + informal_count
            p.formality_score = formal_count / denom if denom else 0.5
        except Exception:
            p.formality_score = 0.5

        # ── Structure ────────────────────────────────────────────────────────
        if all_paragraphs and all_sentences:
            p.avg_paragraph_length = len(all_sentences) / len(all_paragraphs)

        text_lower = all_text.lower()
        transition_hits = sum(1 for tw in TRANSITION_WORDS if tw in text_lower)
        p.transition_word_ratio = transition_hits / len(all_sentences) if all_sentences else 0

        rhetorical_qs = sum(
            1 for s in all_sentences
            if s.strip().endswith("?") and len(s.split()) < 15
        )
        p.rhetorical_question_ratio = rhetorical_qs / len(all_sentences) if all_sentences else 0

        # ── Readability ──────────────────────────────────────────────────────
        try:
            import textstat
            p.flesch_reading_ease = textstat.flesch_reading_ease(all_text[:8000])
            p.gunning_fog = textstat.gunning_fog(all_text[:8000])
        except Exception:
            pass

        # ── Function-word frequencies (Burrows' Delta) ────────────────────────
        p.function_word_freqs = {
            fw: word_counts.get(fw, 0) / total_words
            for fw in FUNCTION_WORDS
        }

        # ── Sample passages for few-shot prompts ──────────────────────────────
        p.sample_passages = self._pick_passages(all_paragraphs)

        return p

    def _pick_passages(self, paragraphs: List[str], n: int = 5, max_words: int = 120) -> List[str]:
        """Pick n representative paragraphs of reasonable length."""
        candidates = [
            p for p in paragraphs
            if 30 <= len(p.split()) <= max_words
        ]
        # Prefer variety — pick evenly spaced ones
        if len(candidates) <= n:
            return candidates
        step = len(candidates) // n
        return [candidates[i * step] for i in range(n)]

    def build_style_description(self, profile: StyleProfile) -> str:
        """Generate a concise qualitative description of the style (no LLM needed)."""
        parts = []

        # Sentence complexity
        asl = profile.avg_sentence_length
        if asl < 12:
            parts.append("short, punchy sentences")
        elif asl < 20:
            parts.append("medium-length sentences with moderate complexity")
        else:
            parts.append("long, complex sentences with multiple clauses")

        # Vocabulary
        if profile.type_token_ratio > 0.6:
            parts.append("a rich and varied vocabulary")
        elif profile.type_token_ratio > 0.4:
            parts.append("a moderately varied vocabulary")
        else:
            parts.append("a repetitive, conversational vocabulary")

        # Tone
        if profile.sentiment_polarity > 0.15:
            parts.append("a generally positive and optimistic tone")
        elif profile.sentiment_polarity < -0.1:
            parts.append("a critical or skeptical tone")
        else:
            parts.append("a neutral, balanced tone")

        # Formality
        if profile.formality_score > 0.65:
            parts.append("formal, academic language")
        elif profile.formality_score > 0.45:
            parts.append("semi-formal language")
        else:
            parts.append("informal, conversational language")

        # Transitions
        if profile.transition_word_ratio > 0.3:
            parts.append("frequent use of transition words to connect ideas")

        # First person
        if profile.first_person_ratio > 0.03:
            parts.append("frequent first-person perspective (I, we)")

        # Hedging
        if profile.hedge_word_ratio > 0.02:
            parts.append("hedging language (perhaps, might, seems)")

        # Punctuation quirks
        if profile.em_dash_per_100 > 0.5:
            parts.append("notable use of em-dashes for emphasis")
        if profile.semicolon_per_100 > 0.3:
            parts.append("semicolons to join related clauses")
        if profile.parenthetical_ratio > 0.15:
            parts.append("parenthetical asides")

        description = "This author writes with " + "; ".join(parts) + "."
        return description
