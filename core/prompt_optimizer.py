"""
Prompt Optimizer — evolves a population of style-replication prompts
through a genetic algorithm to maximize the Style Match Score.

Algorithm per generation:
  1. Evaluate  — run each prompt on the target task, score output
  2. Select    — keep top-N elite prompts
  3. Mutate    — ask a cheap LLM to improve weak prompts
  4. Crossover — combine two elite prompts into a child
  5. Repeat for N generations
"""
import random
import time
from typing import List, Dict, Callable, Optional

from .llm_router import LLMRouter
from .similarity_scorer import SimilarityScorer
from .prompt_generator import PromptGenerator
from .style_analyzer import StyleProfile
import config


class GenerationResult:
    def __init__(self, generation: int):
        self.generation = generation
        self.evaluated: List[Dict] = []   # sorted best-first

    @property
    def best_score(self) -> float:
        return self.evaluated[0]["score"] if self.evaluated else 0.0

    @property
    def best_prompt(self) -> Optional[Dict]:
        return self.evaluated[0] if self.evaluated else None


class PromptOptimizer:
    """Genetic algorithm that evolves prompts to maximise style match."""

    def __init__(
        self,
        llm_router: LLMRouter,
        scorer: SimilarityScorer,
        generator: PromptGenerator,
    ):
        self.llm = llm_router
        self.scorer = scorer
        self.generator = generator

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        profile: StyleProfile,
        writing_task: str,
        generation_model: str,
        generations: int = 4,
        population_size: int = 12,
        elite_fraction: float = 0.33,
        on_progress: Optional[Callable] = None,
    ) -> List[GenerationResult]:
        """
        Run the full GA.

        on_progress(gen_idx, total_gens, result: GenerationResult) — called after
        each generation for live Streamlit updates.

        Returns list of GenerationResult objects, one per generation.
        """
        history: List[GenerationResult] = []

        # ── Gen 0: seed population ────────────────────────────────────────────
        population = self.generator.generate_seed_prompts(
            profile, writing_task, n=population_size
        )

        for gen_idx in range(generations):
            result = GenerationResult(gen_idx)

            # 1. Evaluate every prompt in the population
            result.evaluated = self._evaluate_all(
                population, writing_task, generation_model
            )
            history.append(result)

            if on_progress:
                on_progress(gen_idx, generations, result)

            # Final generation — no need to evolve further
            if gen_idx == generations - 1:
                break

            # 2. Select elite
            n_elite = max(2, int(len(result.evaluated) * elite_fraction))
            elite = result.evaluated[:n_elite]

            # 3. Build next generation
            next_pop = [e["prompt"] for e in elite]

            # Mutate until population is ~70 % full
            target_after_mutate = int(population_size * 0.7)
            while len(next_pop) < target_after_mutate:
                parent = random.choice(elite)
                mutated = self._mutate(parent)
                if mutated:
                    next_pop.append(mutated)

            # Crossover to fill remainder
            while len(next_pop) < population_size:
                if len(elite) >= 2:
                    p1, p2 = random.sample(elite, 2)
                    child = self._crossover(p1, p2)
                    if child:
                        next_pop.append(child)
                else:
                    next_pop.append(random.choice(elite)["prompt"])

            population = next_pop

            # Small delay between generations to be nice to rate limits
            time.sleep(1)

        return history

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_all(
        self, population: List[Dict], task: str, model_key: str
    ) -> List[Dict]:
        """Score every prompt; return sorted list (best first)."""
        results = []
        for prompt in population:
            try:
                output = self.llm.generate(
                    model_key,
                    prompt["system"],
                    prompt["user"] if "user" in prompt else task,
                    max_tokens=900,
                    temperature=0.7,
                )
                scores = self.scorer.score(output)
                results.append({
                    "prompt": prompt,
                    "output": output,
                    "score": scores["composite"],
                    "dim_scores": scores,
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "output": "",
                    "score": 0.0,
                    "dim_scores": {},
                    "error": str(e),
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def _mutate(self, evaluated_prompt: Dict) -> Optional[Dict]:
        """Use a cheap LLM to produce an improved version of a prompt."""
        weak_dims = self.scorer.get_weak_dimensions(evaluated_prompt.get("dim_scores", {}))
        mutation_request = PromptGenerator.build_mutation_request(
            evaluated_prompt["prompt"], weak_dims
        )
        try:
            new_system = self.llm.generate(
                config.GA_MUTATION_MODEL,
                "",
                mutation_request,
                max_tokens=600,
                temperature=0.8,
            )
            if not new_system.strip():
                return None
            mutated = dict(evaluated_prompt["prompt"])
            mutated["system"] = new_system.strip()
            mutated["strategy"] = evaluated_prompt["prompt"]["strategy"] + "_mutated"
            return mutated
        except Exception:
            return None

    # ── Crossover ────────────────────────────────────────────────────────────

    def _crossover(self, parent_a: Dict, parent_b: Dict) -> Optional[Dict]:
        """Combine two parent prompts into a child prompt."""
        crossover_request = PromptGenerator.build_crossover_request(
            parent_a["prompt"], parent_b["prompt"]
        )
        try:
            new_system = self.llm.generate(
                config.GA_MUTATION_MODEL,
                "",
                crossover_request,
                max_tokens=600,
                temperature=0.7,
            )
            if not new_system.strip():
                return None
            child = {
                "system": new_system.strip(),
                "user": parent_a["prompt"].get("user", ""),
                "strategy": "crossover",
                "id": f"child_{random.randint(1000, 9999)}",
            }
            return child
        except Exception:
            return None
