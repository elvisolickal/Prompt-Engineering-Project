"""
AI Writing Style Engine — Streamlit Dashboard
"""
import sys, os, json, uuid, time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
import config
from core.ingestion import CorpusIngester
from core.style_analyzer import StyleAnalyzer, StyleProfile
from core.similarity_scorer import SimilarityScorer
from core.prompt_generator import PromptGenerator
from core.llm_router import LLMRouter
from core.prompt_optimizer import PromptOptimizer
from core.generation_engine import GenerationEngine
from core.results_store import ResultsStore

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Writing Style Engine",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22 !important; }
.score-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}
.score-value { font-size: 2.5rem; font-weight: 700; color: #6ee7b7; }
.score-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin: 10px 0;
}
.best-badge {
    background: linear-gradient(90deg, #059669, #10b981);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
h1, h2, h3 { color: #f0f6fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        documents=[],
        profile=None,
        corpus_text="",
        scorer=None,
        author_name="Author",
        opt_history=[],
        best_prompts=[],
        gen_results=[],
        run_id=str(uuid.uuid4())[:8],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Singletons ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_store():
    return ResultsStore(config.DB_PATH)

@st.cache_resource
def get_ingester():
    return CorpusIngester()

@st.cache_resource
def get_analyzer():
    return StyleAnalyzer()

@st.cache_resource
def get_llm():
    return LLMRouter()

store    = get_store()
ingester = get_ingester()
analyzer = get_analyzer()
llm      = get_llm()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Writing Style Engine")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📁 Setup & Upload", "🎨 Style Profile", "🧬 Optimize Prompts",
         "✍️ Generate Content", "🏆 Leaderboard", "📤 Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**API Status**")
    st.markdown(f"{'🟢' if config.OPENAI_API_KEY else '🔴'} OpenAI")
    st.markdown(f"{'🟢' if config.GEMINI_API_KEY else '🔴'} Gemini")
    available_models = llm.list_available_models()
    if st.session_state.profile:
        st.markdown("---")
        st.success(f"Profile: **{st.session_state.author_name}**\n\n"
                   f"{st.session_state.profile.total_words:,} words • "
                   f"{st.session_state.profile.total_documents} docs")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Setup & Upload
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📁 Setup & Upload":
    st.title("📁 Setup & Upload Writing Samples")
    st.markdown("Upload past writing to build an author's style fingerprint.")

    col1, col2 = st.columns([1, 1])
    with col1:
        author_name = st.text_input("Author Name", value=st.session_state.author_name)
        st.session_state.author_name = author_name

    st.markdown("### Upload Files")
    uploaded = st.file_uploader(
        "Drop files here (.txt, .md, .pdf, .docx, .eml)",
        accept_multiple_files=True,
        type=["txt", "md", "pdf", "docx", "eml"],
    )

    st.markdown("### Or Paste Text")
    pasted = st.text_area("Paste writing sample here", height=200, key="paste_area")
    paste_type = st.selectbox("Content type", ["essay", "article", "email", "social", "other"])

    if st.button("➕ Add Pasted Text") and pasted.strip():
        doc = ingester.load_from_text(pasted, "pasted_text", paste_type)
        if doc:
            st.session_state.documents.append(doc)
            st.success(f"Added {doc.word_count:,} words.")
        else:
            st.warning("Text too short (< 80 words). Please add more content.")

    if uploaded:
        for f in uploaded:
            data = f.read()
            doc = ingester.load_uploaded_bytes(data, f.name)
            if doc and doc not in st.session_state.documents:
                st.session_state.documents.append(doc)

    docs = st.session_state.documents
    if docs:
        total_words = ingester.get_total_words(docs)
        st.markdown(f"### Corpus: {len(docs)} document(s) · {total_words:,} words")
        for d in docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"📄 **{d.filename}** — {d.word_count:,} words ({d.source_type})")
            if c2.button("✕", key=f"del_{d.filename}"):
                st.session_state.documents = [x for x in docs if x.filename != d.filename]
                st.rerun()

        if total_words < 500:
            st.warning("⚠️ Add more text (at least 500 words) for accurate style analysis.")

        if st.button("🔬 Analyze Style & Build Profile", type="primary"):
            with st.spinner("Extracting style features…"):
                corpus_text = ingester.get_corpus_text(docs)
                profile = analyzer.analyze(docs, author_name)
                profile.style_description = analyzer.build_style_description(profile)

                scorer = SimilarityScorer()
                scorer.fit(profile, corpus_text)

                st.session_state.profile = profile
                st.session_state.corpus_text = corpus_text
                st.session_state.scorer = scorer
            st.success("✅ Style profile built! Go to 🎨 Style Profile to explore it.")
    else:
        st.info("Upload writing samples or paste text above to get started.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Style Profile
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Style Profile":
    st.title("🎨 Style Profile")

    if not st.session_state.profile:
        st.warning("No profile yet. Go to 📁 Setup & Upload first.")
        st.stop()

    p: StyleProfile = st.session_state.profile

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="score-card"><div class="score-value">{p.total_words:,}</div><div class="score-label">Total Words</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="score-card"><div class="score-value">{p.avg_sentence_length:.1f}</div><div class="score-label">Avg Sentence Length</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="score-card"><div class="score-value">{p.type_token_ratio:.2f}</div><div class="score-label">Vocab Richness (TTR)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="score-card"><div class="score-value">{p.flesch_reading_ease:.0f}</div><div class="score-label">Flesch Readability</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Style Radar")
        cats = ["Vocab Richness", "Sentence Complexity", "Formality",
                "Positivity", "Hedge Language", "Transition Words", "Punctuation"]
        vals = [
            min(p.type_token_ratio * 2, 1.0),
            min(p.avg_sentence_length / 30, 1.0),
            p.formality_score,
            max(0, p.sentiment_polarity + 0.5),
            min(p.hedge_word_ratio * 40, 1.0),
            min(p.transition_word_ratio, 1.0),
            min((p.comma_per_100 + p.semicolon_per_100 + p.em_dash_per_100) / 5, 1.0),
        ]
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(99,102,241,0.25)",
            line=dict(color="#6366f1", width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#9ca3af"))),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#f0f6fc"),
            margin=dict(l=40, r=40, t=40, b=40),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Style Description")
        st.info(p.style_description)

        st.subheader("Feature Breakdown")
        feat_data = {
            "Feature": ["Avg Word Length", "Sentence Length Var.", "Long Sent. Ratio",
                        "Comma/100 words", "Em-Dash/100", "1st Person Ratio",
                        "Gunning Fog", "Avg Paragraph Len"],
            "Value": [f"{p.avg_word_length:.2f}", f"{p.sentence_length_variance:.1f}",
                      f"{p.long_sentence_ratio:.0%}", f"{p.comma_per_100:.2f}",
                      f"{p.em_dash_per_100:.2f}", f"{p.first_person_ratio:.2%}",
                      f"{p.gunning_fog:.1f}", f"{p.avg_paragraph_length:.1f} sents"],
        }
        st.dataframe(pd.DataFrame(feat_data), hide_index=True, use_container_width=True)

    if p.sample_passages:
        st.markdown("---")
        st.subheader("Sample Passages (used for few-shot prompts)")
        for i, passage in enumerate(p.sample_passages[:3]):
            with st.expander(f"Passage {i+1} ({len(passage.split())} words)"):
                st.write(passage)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Optimize Prompts
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Optimize Prompts":
    st.title("🧬 Genetic Prompt Optimizer")

    if not st.session_state.profile:
        st.warning("No profile yet. Go to 📁 Setup & Upload first.")
        st.stop()

    if not available_models:
        st.error("No API keys configured. Add keys to your .env file.")
        st.stop()

    st.markdown("The optimizer evolves prompts across generations to maximize Style Match Score.")

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_task = st.text_area("Writing Task", height=100,
            value="Write a 3-paragraph essay about the importance of critical thinking in modern society.",
            key="opt_task")
    with col2:
        eval_model = st.selectbox("Evaluation Model", available_models)
        generations = st.slider("Generations", 1, 6, 3)
    with col3:
        pop_size = st.slider("Population Size", 4, 16, 8)
        elite_frac = st.slider("Elite Fraction", 0.2, 0.5, 0.33)

    st.info(f"Estimated API calls: ~{pop_size * generations + (pop_size * (1-elite_frac) * generations):.0f} "
            f"(evaluation + mutation/crossover)")

    if st.button("🚀 Run Optimizer", type="primary"):
        if not opt_task.strip():
            st.error("Please enter a writing task.")
        else:
            prog_bar    = st.progress(0)
            gen_status  = st.empty()
            chart_ph    = st.empty()
            scores_over_time = []

            def on_progress(gen_idx, total, result):
                pct = (gen_idx + 1) / total
                prog_bar.progress(pct)
                gen_status.markdown(
                    f"**Generation {gen_idx + 1}/{total}** — "
                    f"Best score: **{result.best_score:.1f}**"
                )
                scores_over_time.append({
                    "Generation": gen_idx + 1,
                    "Best Score": result.best_score,
                    "Avg Score": sum(e["score"] for e in result.evaluated) / max(len(result.evaluated), 1),
                })
                df_prog = pd.DataFrame(scores_over_time)
                fig = px.line(df_prog, x="Generation", y=["Best Score", "Avg Score"],
                              template="plotly_dark", markers=True,
                              color_discrete_map={"Best Score": "#6ee7b7", "Avg Score": "#818cf8"})
                fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=280)
                chart_ph.plotly_chart(fig, use_container_width=True)

            scorer = st.session_state.scorer
            generator = PromptGenerator()
            optimizer = PromptOptimizer(llm, scorer, generator)

            try:
                with st.spinner("Evolving prompts…"):
                    history = optimizer.run(
                        st.session_state.profile,
                        opt_task,
                        generation_model=eval_model,
                        generations=generations,
                        population_size=pop_size,
                        elite_fraction=elite_frac,
                        on_progress=on_progress,
                    )

                st.session_state.opt_history = history
                all_results = [item for gen in history for item in gen.evaluated]
                best = sorted(all_results, key=lambda x: x["score"], reverse=True)
                st.session_state.best_prompts = [r["prompt"] for r in best[:5]]

                # Save to DB
                for gen_result in history:
                    store.save_results_batch(
                        gen_result.evaluated, st.session_state.run_id,
                        st.session_state.author_name, opt_task, gen_result.generation
                    )

                prog_bar.progress(1.0)
                gen_status.success(f"✅ Optimization complete! Best score: **{best[0]['score']:.1f}/100**")

                st.markdown("### 🏅 Top Evolved Prompts")
                for i, r in enumerate(best[:5]):
                    with st.expander(f"#{i+1} Score {r['score']:.1f} — {r['prompt'].get('strategy','')}", expanded=(i==0)):
                        st.markdown("**System Prompt:**")
                        st.code(r["prompt"]["system"], language="text")
                        st.markdown("**Generated Output:**")
                        st.write(r["output"])
                        dim = r.get("dim_scores", {})
                        d1, d2, d3, d4, d5 = st.columns(5)
                        d1.metric("Feature Dist", f"{dim.get('feature_dist', 0)*100:.0f}")
                        d2.metric("Burrows Δ", f"{dim.get('burrows_delta', 0)*100:.0f}")
                        d3.metric("Embedding", f"{dim.get('embedding', 0)*100:.0f}")
                        d4.metric("TF-IDF", f"{dim.get('tfidf', 0)*100:.0f}")
                        d5.metric("Readability", f"{dim.get('readability', 0)*100:.0f}")

            except Exception as e:
                st.error(f"Optimization failed: {e}")

    elif st.session_state.opt_history:
        st.success(f"Previous run: {len(st.session_state.best_prompts)} best prompts ready.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Generate Content
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "✍️ Generate Content":
    st.title("✍️ Generate Content")

    if not st.session_state.profile:
        st.warning("No profile yet. Go to 📁 Setup & Upload first.")
        st.stop()

    if not available_models:
        st.error("No API keys configured.")
        st.stop()

    gen_task = st.text_area("Writing Task", height=100, key="gen_task",
        value="Write a 3-paragraph essay about the role of artificial intelligence in education.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_models = st.multiselect("Models", available_models, default=available_models[:2])
    with col2:
        max_tokens = st.slider("Max Tokens", 300, 1500, 800)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    with col3:
        use_best = st.checkbox("Use best evolved prompts", value=bool(st.session_state.best_prompts))
        prompt_mode = st.selectbox("Prompt strategy (if no best prompts)", ["hybrid", "few_shot", "feature_explicit", "persona"])

    if st.button("⚡ Generate", type="primary"):
        if not gen_task.strip():
            st.error("Please enter a task.")
        elif not sel_models:
            st.error("Select at least one model.")
        else:
            scorer = st.session_state.scorer
            engine = GenerationEngine(llm, scorer)
            generator = PromptGenerator()

            if use_best and st.session_state.best_prompts:
                prompts = st.session_state.best_prompts[:3]
            else:
                all_seeds = generator.generate_seed_prompts(st.session_state.profile, gen_task)
                prompts = [p for p in all_seeds if p["strategy"] == prompt_mode][:2]
                if not prompts:
                    prompts = all_seeds[:2]

            results_ph = st.empty()
            live_results = []

            def on_result(r):
                live_results.append(r)
                results_ph.markdown(f"Generated {len(live_results)} output(s)…")

            with st.spinner("Generating…"):
                results = engine.run_matrix(
                    gen_task, prompts, sel_models,
                    max_tokens=max_tokens, temperature=temperature,
                    on_result=on_result,
                )

            store.save_results_batch(results, st.session_state.run_id,
                                     st.session_state.author_name, gen_task)
            st.session_state.gen_results = results

            results_ph.empty()
            st.markdown("### Results — ranked by Style Match Score")
            for i, r in enumerate(results):
                badge = '<span class="best-badge">BEST MATCH</span>' if i == 0 else ""
                st.markdown(
                    f'<div class="result-card">'
                    f'<b>{r["model"]}</b> · {r["strategy"]} {badge}'
                    f' — <b style="color:#6ee7b7">{r["score"]:.1f}/100</b>'
                    f'</div>', unsafe_allow_html=True
                )
                with st.expander(f"View output — Score {r['score']:.1f}", expanded=(i == 0)):
                    st.write(r["output"] or "_No output generated_")
                    if r.get("error"):
                        st.error(r["error"])
                    dim = r.get("dim_scores", {})
                    if dim:
                        d1, d2, d3, d4, d5 = st.columns(5)
                        d1.metric("Feature Dist", f"{dim.get('feature_dist',0)*100:.0f}")
                        d2.metric("Burrows Δ", f"{dim.get('burrows_delta',0)*100:.0f}")
                        d3.metric("Embedding", f"{dim.get('embedding',0)*100:.0f}")
                        d4.metric("TF-IDF", f"{dim.get('tfidf',0)*100:.0f}")
                        d5.metric("Readability", f"{dim.get('readability',0)*100:.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Leaderboard":
    st.title("🏆 Leaderboard")

    authors = store.get_all_authors()
    if not authors:
        st.info("No experiments yet. Run the optimizer or generator first.")
        st.stop()

    sel_author = st.selectbox("Filter by Author", ["All"] + authors)
    author_filter = sel_author if sel_author != "All" else None

    data = store.get_leaderboard(author_filter, limit=100)
    if not data:
        st.info("No results found.")
        st.stop()

    df = pd.DataFrame(data)
    cols_show = ["author_name", "model", "strategy", "score", "score_feature_dist",
                 "score_burrows", "score_embedding", "score_tfidf", "created_at"]
    df_show = df[[c for c in cols_show if c in df.columns]].copy()
    df_show.rename(columns={
        "author_name": "Author", "model": "Model", "strategy": "Strategy",
        "score": "Score", "score_feature_dist": "Feature Dist",
        "score_burrows": "Burrows Δ", "score_embedding": "Embedding",
        "score_tfidf": "TF-IDF", "created_at": "Date",
    }, inplace=True)

    st.dataframe(df_show.style.format({
        "Score": "{:.1f}", "Feature Dist": "{:.2f}", "Burrows Δ": "{:.2f}",
        "Embedding": "{:.2f}", "TF-IDF": "{:.2f}",
    }), use_container_width=True, height=400)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution by Model")
        if "model" in df.columns and "score" in df.columns:
            fig = px.box(df, x="model", y="score", template="plotly_dark",
                         color="model", title="Score Distribution per Model")
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Score by Strategy")
        if "strategy" in df.columns:
            strat_avg = df.groupby("strategy")["score"].mean().reset_index().sort_values("score", ascending=False)
            fig2 = px.bar(strat_avg, x="strategy", y="score", template="plotly_dark",
                          color="score", color_continuous_scale="Viridis")
            fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
            st.plotly_chart(fig2, use_container_width=True)

    if author_filter:
        model_comp = store.get_model_comparison(author_filter)
        if model_comp:
            st.subheader("Model Comparison")
            st.dataframe(pd.DataFrame(model_comp), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Export
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Export":
    st.title("📤 Export")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Style Profile")
        if st.session_state.profile:
            profile_json = json.dumps(st.session_state.profile.to_dict(), indent=2, default=str)
            st.download_button("⬇ Download Style Profile (JSON)", profile_json,
                               f"{st.session_state.author_name}_style_profile.json", "application/json")
        else:
            st.info("No profile built yet.")

    with col2:
        st.subheader("Export Best Prompts")
        if st.session_state.best_prompts:
            prompts_md = f"# Best Prompts for {st.session_state.author_name}\n\n"
            for i, pr in enumerate(st.session_state.best_prompts, 1):
                prompts_md += f"## Prompt #{i} — {pr.get('strategy','')}\n\n"
                prompts_md += f"### System Prompt\n```\n{pr.get('system','')}\n```\n\n"
            st.download_button("⬇ Download Best Prompts (Markdown)", prompts_md,
                               f"{st.session_state.author_name}_best_prompts.md", "text/markdown")
        else:
            st.info("Run the optimizer first to get evolved prompts.")

    st.subheader("Export Leaderboard")
    authors = store.get_all_authors()
    if authors:
        export_author = st.selectbox("Author", ["All"] + authors, key="export_author")
        data = store.get_leaderboard(export_author if export_author != "All" else None, limit=500)
        if data:
            df_export = pd.DataFrame(data)
            csv = df_export.to_csv(index=False)
            st.download_button("⬇ Download Leaderboard (CSV)", csv,
                               "leaderboard_export.csv", "text/csv")
    else:
        st.info("No experiment data yet.")
