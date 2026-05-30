"""
Generation Engine — run a writing task across multiple models and prompts,
collect outputs, and score them against the author's StyleProfile.
"""
from typing import List, Dict, Optional, Callable
from .llm_router import LLMRouter
from .similarity_scorer import SimilarityScorer


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

        on_result(result_dict) — called after each generation for live updates.
        Returns list of result dicts sorted by score descending.
        """
        results = []

        for model_key in models:
            for prompt in prompts:
                result = self._run_one(
                    model_key, prompt, writing_task, max_tokens, temperature
                )
                results.append(result)
                if on_result:
                    on_result(result)

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
