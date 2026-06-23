"""
Generation Engine — run a writing task across multiple models and prompts,
collect outputs, and score them against the author's StyleProfile.
Includes rate-limit-safe delays between API calls.
"""
import time
from typing import List, Dict, Optional, Callable
from .llm_router import LLMRouter
from .similarity_scorer import SimilarityScorer

# Minimum seconds to wait between API calls to avoid rate limits
_CALL_DELAY = 3.0


class GenerationEngine:
    """Run the same task across model × prompt combinations and score results."""

    def __init__(self, llm_router: LLMRouter, scorer: SimilarityScorer):
        self.llm = llm_router
        self.scorer = scorer

    def run_matrix(
        self,
        writing_task: str,
        prompts: List[Dict],
        models: List[str],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        on_result: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Run every (model, prompt) combination and return scored results.
        Sleeps between calls to respect free-tier rate limits.
        Returns list of result dicts sorted by score descending.
        """
        results = []

        for i, model_key in enumerate(models):
            for j, prompt in enumerate(prompts):
                result = self._run_one(
                    model_key, prompt, writing_task, max_tokens, temperature
                )
                results.append(result)
                if on_result:
                    on_result(result)
                # Rate-limit pause between calls (skip after last call)
                if not (i == len(models) - 1 and j == len(prompts) - 1):
                    time.sleep(_CALL_DELAY)

        return sorted(results, key=lambda r: r["score"], reverse=True)

    def _run_one(
        self,
        model_key: str,
        prompt: Dict,
        task: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict:
        try:
            output = self.llm.generate(
                model_key,
                prompt.get("system", ""),
                prompt.get("user", task),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            scores = self.scorer.score(output)
            return {
                "model": model_key,
                "strategy": prompt.get("strategy", "unknown"),
                "prompt_id": prompt.get("id", ""),
                "system_prompt": prompt.get("system", ""),
                "output": output,
                "score": scores.get("composite", 0.0),
                "dim_scores": scores,
                "error": None,
            }
        except Exception as e:
            return {
                "model": model_key,
                "strategy": prompt.get("strategy", "unknown"),
                "prompt_id": prompt.get("id", ""),
                "system_prompt": prompt.get("system", ""),
                "output": "",
                "score": 0.0,
                "dim_scores": {},
                "error": str(e),
            }

    def generate_single(
        self,
        model_key: str,
        system_prompt: str,
        task: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Dict:
        """Quick single-shot generation with scoring."""
        prompt = {"system": system_prompt, "user": task, "strategy": "custom", "id": "custom"}
        return self._run_one(model_key, prompt, task, max_tokens, temperature)
