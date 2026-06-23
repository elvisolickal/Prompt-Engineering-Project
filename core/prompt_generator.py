"""
Prompt Generator — create a diverse seed population of prompts
that instruct an LLM to write in the author's style.

Seven strategy archetypes:
  1. bare_instruction  — minimal one-liner
  2. feature_explicit  — lists top measured style features
  3. few_shot          — includes actual sample passages
  4. persona           — qualitative voice description
  5. chain_of_thought  — ask the model to reason before writing
  6. contrastive       — what NOT to do as well as what TO do
  7. hybrid            — few-shot + explicit features (strongest baseline)
"""
from typing import List, Dict
from .style_analyzer import StyleProfile


class PromptGenerator:
    """Generate a population of diverse style-replication prompts."""

    def generate_seed_prompts(
        self,
        profile: StyleProfile,
        writing_task: str,
        n: int = 12,
    ) -> List[Dict]:
        """
        Return a list of prompt dicts:
          {
            "id": str,
            "strategy": str,
            "system": str,    # system-level instruction
            "user": str,      # user message (includes the task)
          }
        """
        prompts = []
        prompts += self._bare_instruction(profile, writing_task)
        prompts += self._feature_explicit(profile, writing_task)
        prompts += self._few_shot(profile, writing_task)
        prompts += self._persona(profile, writing_task)
        prompts += self._chain_of_thought(profile, writing_task)
        prompts += self._contrastive(profile, writing_task)
        prompts += self._hybrid(profile, writing_task)

        # Trim or pad to exactly n
        prompts = prompts[:n]
        for i, p in enumerate(prompts):
            p["id"] = f"seed_{i:03d}"
        return prompts

    def get_prompt_preview(self, strategy: str, profile: StyleProfile, task: str) -> str:
        """Return the full system prompt text for a given strategy name."""
        mapping = {
            "bare_instruction":  lambda: self._bare_instruction(profile, task),
            "bare_instruction_v2": lambda: self._bare_instruction(profile, task),
            "feature_explicit":  lambda: self._feature_explicit(profile, task),
            "few_shot":          lambda: self._few_shot(profile, task),
            "persona":           lambda: self._persona(profile, task),
            "chain_of_thought":  lambda: self._chain_of_thought(profile, task),
            "contrastive":       lambda: self._contrastive(profile, task),
            "hybrid":            lambda: self._hybrid(profile, task),
            "hybrid_rich":       lambda: self._hybrid(profile, task),
        }
        fn = mapping.get(strategy)
        if fn:
            results = fn()
            # For hybrid_rich pick the second result, otherwise first
            idx = 1 if strategy == "hybrid_rich" and len(results) > 1 else 0
            return results[idx]["system"] if results else "(no prompt generated for this strategy)"
        return f"(unknown strategy: {strategy!r})"

    # ── Strategy builders ─────────────────────────────────────────────────────

    def _bare_instruction(self, p: StyleProfile, task: str) -> List[Dict]:
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "bare_instruction",
                "system": (
                    f"You are {p.author_name}. Write exactly as this person writes.\n{suffix}"
                ),
                "user": task,
            },
            {
                "strategy": "bare_instruction_v2",
                "system": (
                    f"Mimic the writing style of {p.author_name} as closely as possible. "
                    f"Do not break character.\n{suffix}"
                ),
                "user": task,
            },
        ]

    def _feature_explicit(self, p: StyleProfile, task: str) -> List[Dict]:
        features = self._build_feature_list(p)
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "feature_explicit",
                "system": (
                    f"Write in the style of {p.author_name}. "
                    f"Apply these specific stylistic properties:\n{features}\n\n{suffix}"
                ),
                "user": task,
            }
        ]

    def _few_shot(self, p: StyleProfile, task: str) -> List[Dict]:
        if not p.sample_passages:
            return []
        examples = "\n\n---\n\n".join(p.sample_passages[:3])
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "few_shot",
                "system": (
                    f"Study the following writing samples by {p.author_name}, "
                    "then complete the task using the exact same voice, rhythm, and style.\n\n"
                    f"WRITING SAMPLES:\n\n{examples}\n\n{suffix}"
                ),
                "user": task,
            }
        ]

    def _persona(self, p: StyleProfile, task: str) -> List[Dict]:
        desc = p.style_description or self._heuristic_description(p)
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "persona",
                "system": (
                    f"You are channeling the voice of {p.author_name}. "
                    f"{desc} "
                    f"Maintain this voice faithfully throughout your response.\n\n{suffix}"
                ),
                "user": task,
            }
        ]

    def _chain_of_thought(self, p: StyleProfile, task: str) -> List[Dict]:
        features = self._build_feature_list(p)
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "chain_of_thought",
                "system": (
                    f"You will write in the style of {p.author_name}. "
                    "Before writing, briefly reason (in a hidden scratchpad) about "
                    "which stylistic elements to apply:\n"
                    f"{features}\n\n"
                    f"Then produce only the final piece — no reasoning in the output.\n\n{suffix}"
                ),
                "user": task,
            }
        ]

    def _contrastive(self, p: StyleProfile, task: str) -> List[Dict]:
        dos, donts = self._build_dos_and_donts(p)
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "contrastive",
                "system": (
                    f"Write in the style of {p.author_name}.\n\n"
                    f"DO:\n{dos}\n\nDO NOT:\n{donts}\n\n{suffix}"
                ),
                "user": task,
            }
        ]

    def _hybrid(self, p: StyleProfile, task: str) -> List[Dict]:
        if not p.sample_passages:
            return []
        features = self._build_feature_list(p)
        examples = "\n\n---\n\n".join(p.sample_passages[:2])
        suffix = self._anti_ai_suffix(p)
        return [
            {
                "strategy": "hybrid",
                "system": (
                    f"Replicate the voice of {p.author_name} precisely.\n\n"
                    f"STYLE FEATURES TO APPLY:\n{features}\n\n"
                    f"EXAMPLE PASSAGES FROM {p.author_name.upper()}:\n\n{examples}\n\n"
                    f"Now write in this exact style.\n\n{suffix}"
                ),
                "user": task,
            },
            {
                "strategy": "hybrid_rich",
                "system": (
                    f"You are a professional ghostwriter tasked with perfectly imitating "
                    f"{p.author_name}'s writing style.\n\n"
                    f"MEASURABLE STYLE ATTRIBUTES:\n{features}\n\n"
                    f"AUTHENTIC WRITING SAMPLES:\n\n{examples}\n\n"
                    "Your output must be indistinguishable from something "
                    f"{p.author_name} actually wrote.\n\n{suffix}"
                ),
                "user": task,
            },
        ]

    # ── Anti-AI-detection suffix ───────────────────────────────────────────────

    def _anti_ai_suffix(self, p: StyleProfile) -> str:
        """Generate instructions that discourage AI-sounding patterns."""
        lines = [
            "CRITICAL AUTHENTICITY RULES (follow strictly):",
            "• Do NOT start with 'Certainly', 'Of course', 'Absolutely', 'Great question', or any sycophantic opener.",
            "• Do NOT produce a symmetrically structured response with a clean intro-body-conclusion arc unless that matches the samples.",
            "• Do NOT use bullet points or numbered lists unless the author's samples contain them.",
            "• Vary sentence lengths naturally — mix short punchy sentences with longer flowing ones as the author does.",
            "• Avoid overly polished or grammatically perfect prose; if the author uses fragments, run-ons, or dashes — use them.",
            "• Do NOT end with a tidy summarizing sentence that wraps everything up neatly — that is an AI habit.",
            "• Do NOT use filler transition phrases like 'It is worth noting', 'In today\'s world', 'In conclusion'.",
        ]
        if p.first_person_ratio < 0.02:
            lines.append("• Write in third person or second person as the author prefers — avoid heavy 'I' usage.")
        if p.em_dash_per_100 < 0.2:
            lines.append("• Avoid em-dashes; the author does not use them frequently.")
        if p.semicolon_per_100 < 0.1:
            lines.append("• Avoid semicolons; the author rarely uses them.")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_feature_list(self, p: StyleProfile) -> str:
        lines = []

        # Sentence length
        asl = p.avg_sentence_length
        if asl > 0:
            lines.append(f"• Average sentence length: {asl:.1f} words")
        if p.long_sentence_ratio > 0.3:
            lines.append("• Frequently uses long, complex sentences (>25 words)")
        if p.short_sentence_ratio > 0.3:
            lines.append("• Frequently uses short, punchy sentences (<8 words)")

        # Vocabulary
        if p.type_token_ratio > 0:
            richness = "rich" if p.type_token_ratio > 0.5 else "moderate" if p.type_token_ratio > 0.35 else "repetitive"
            lines.append(f"• Vocabulary richness: {richness} (TTR={p.type_token_ratio:.2f})")
        if p.rare_word_ratio > 0.1:
            lines.append("• Uses advanced/rare vocabulary regularly")

        # Tone
        if p.sentiment_polarity > 0.15:
            lines.append("• Consistently positive, upbeat tone")
        elif p.sentiment_polarity < -0.1:
            lines.append("• Skeptical, critical, or analytical tone")
        else:
            lines.append("• Neutral, balanced tone")

        if p.formality_score > 0.65:
            lines.append("• Formal, academic register")
        elif p.formality_score < 0.4:
            lines.append("• Informal, conversational register")

        if p.first_person_ratio > 0.03:
            lines.append(f"• Heavy first-person usage (I/we — {p.first_person_ratio*100:.1f}% of words)")
        if p.hedge_word_ratio > 0.015:
            lines.append("• Frequent hedging language (perhaps, might, suggests)")

        # Punctuation
        if p.em_dash_per_100 > 0.4:
            lines.append("• Distinctive em-dash usage for parenthetical emphasis")
        if p.semicolon_per_100 > 0.2:
            lines.append("• Uses semicolons to link closely related clauses")
        if p.parenthetical_ratio > 0.1:
            lines.append("• Frequent parenthetical asides (in parentheses)")
        if p.ellipsis_per_100 > 0.1:
            lines.append("• Uses ellipses for rhetorical effect")

        # Structure
        if p.transition_word_ratio > 0.25:
            lines.append("• Heavy use of transition words (however, therefore, moreover)")
        if p.avg_paragraph_length > 5:
            lines.append(f"• Long paragraphs ({p.avg_paragraph_length:.1f} sentences avg)")
        elif p.avg_paragraph_length < 2.5:
            lines.append("• Short paragraphs — punchy, one-idea-per-paragraph style")

        # Readability
        if p.flesch_reading_ease > 0:
            lines.append(f"• Readability: Flesch {p.flesch_reading_ease:.0f}/100, Gunning Fog {p.gunning_fog:.1f}")

        return "\n".join(lines) if lines else "• Match the author's natural writing style"

    def _build_dos_and_donts(self, p: StyleProfile):
        dos = []
        donts = []

        if p.avg_sentence_length > 20:
            dos.append("Write long, flowing sentences with multiple clauses")
            donts.append("Avoid short, choppy sentences")
        else:
            dos.append("Keep sentences tight and direct")
            donts.append("Avoid rambling, overly complex sentence structures")

        if p.formality_score > 0.6:
            dos.append("Use formal, precise language")
            donts.append("Do not use slang, contractions, or casual phrases")
        else:
            dos.append("Use casual, conversational language")
            donts.append("Avoid stiff, overly formal academic prose")

        if p.first_person_ratio > 0.03:
            dos.append("Write in first person (I, we, my)")
        else:
            donts.append("Avoid first-person — keep an objective voice")

        if p.hedge_word_ratio > 0.015:
            dos.append("Use hedging language to qualify statements")
        else:
            donts.append("Avoid excessive hedging — be direct and assertive")

        if p.em_dash_per_100 > 0.4:
            dos.append("Use em-dashes — like this — to add emphasis or asides")

        dos_str = "\n".join(f"✓ {d}" for d in dos)
        donts_str = "\n".join(f"✗ {d}" for d in donts)
        return dos_str, donts_str

    def _heuristic_description(self, p: StyleProfile) -> str:
        """Fallback description if profile.style_description is empty."""
        from .style_analyzer import StyleAnalyzer
        return StyleAnalyzer().build_style_description(p)

    # ── Mutation / crossover helpers (called by optimizer) ────────────────────

    @staticmethod
    def build_mutation_request(prompt: Dict, weak_dims: List[str]) -> str:
        """Build the LLM instruction to mutate a prompt targeting weak dimensions."""
        dims_str = " and ".join(weak_dims)
        return (
            f"You are a prompt engineering expert specializing in writing-style imitation.\n\n"
            f"This prompt does not yet achieve a good match in: {dims_str}.\n\n"
            f"CURRENT SYSTEM PROMPT:\n{prompt['system']}\n\n"
            f"Rewrite the system prompt to specifically address the weakness in {dims_str}. "
            "Keep everything else intact. Return ONLY the improved system prompt text."
        )

    @staticmethod
    def build_crossover_request(parent_a: Dict, parent_b: Dict) -> str:
        """Build the LLM instruction to crossover two prompts."""
        return (
            "You are a prompt engineering expert. Combine the strongest elements "
            "from these two writing-style prompts into one superior prompt.\n\n"
            f"PROMPT A:\n{parent_a['system']}\n\n"
            f"PROMPT B:\n{parent_b['system']}\n\n"
            "Return ONLY the combined system prompt text, nothing else."
        )
