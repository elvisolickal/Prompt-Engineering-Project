# ✍️ AI Writing Style Engine

An advanced, multi-model AI system that learns a person’s writing style from their past writing samples (essays, messages, articles, emails, or social media posts), extracts measurable linguistic features, and automatically discovers the most effective prompts to replicate their authentic voice using evolutionary genetic algorithms.

---

## 🚀 Key Features

*   **Linguistic Ingestion & Parsing**: Parses `.txt`, `.md`, `.docx`, `.pdf`, and `.eml` files, stripping out signatures and boilerplate.
*   **Stylometric Analysis (25+ Features)**: Measures lexical richness (TTR, Hapax Legomena), syntactic variety (sentence length distributions), punctuation habits (em-dash/semicolon usage frequency), tone (sentiment/subjectivity), readability level (Flesch/Gunning Fog), and first-person pronoun ratios.
*   **Genetic Prompt Optimization**: Evolves a population of prompt variations over multiple generations. Uses a fast LLM-driven evaluator to mutate prompts targeting stylometric weaknesses.
*   **Multi-Model Execution Matrix**: Generates the same task across OpenAI (GPT-4o, GPT-4o-mini) and Google Gemini (1.5 Pro, 1.5 Flash) simultaneously.
*   **Composite Stylometric Scorer**: Evaluates outputs mathematically using an integrated 5-metric score (Feature Vector distance, Burrows' Delta, Cosine semantic embedding similarity, TF-IDF n-grams, and Readability alignment).
*   **Streamlit Web Interface**: A premium dark-mode web dashboard featuring live optimization graphs, radar charts of extracted style fingerprints, and side-by-side model outputs ranked on a leaderboard.

---

## 🛠️ Architecture

```
writing-style-engine/
├── app.py                     # Streamlit entry point (6 tabs)
├── config.py                  # API keys, weights, and GA parameters
├── requirements.txt           # Project dependencies
├── .gitignore                 # Safe defaults preventing .env and database uploads
└── core/
    ├── ingestion.py          # Document loader and tokenizer
    ├── style_analyzer.py     # 25+ stylometric dimension extractor
    ├── similarity_scorer.py  # Multi-metric composite similarity engine
    ├── prompt_generator.py   # Seed prompt strategies & crossover/mutation builders
    ├── llm_router.py         # Unified OpenAI + Gemini API wrapper with rate-limit retries
    ├── prompt_optimizer.py   # Genetic evolutionary algorithm orchestrator
    ├── generation_engine.py  # Matrix runner for model × prompt evaluations
    └── results_store.py      # SQLite experiment persistence layer (SQLAlchemy Core)
```

---

## ⚙️ Installation & Setup

### 1. Clone & Set active workspace
Open your favorite terminal and navigate to the project directory:
```bash
cd "C:\Users\irits\.gemini\antigravity\scratch\writing-style-engine"
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys:
```env
OPENAI_API_KEY=sk-proj-YourOpenAiKeyHere...
GEMINI_API_KEY=AIzaSyYourGeminiKeyHere...
```

### 3. Run the Streamlit Dashboard
Launch the app locally:
```bash
python -m streamlit run app.py
```
*(If your system uses a dedicated installation of Python, run:)*
```bash
C:\Users\irits\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧬 How to Use the System

1.  **Ingestion & Profile Creation (`📁 Setup & Upload` page)**:
    *   Upload your sample documents or paste text directly.
    *   Click **Analyze Style** to extract your personalized `StyleProfile`.
2.  **Visualizing the Fingerprint (`🎨 Style Profile` page)**:
    *   Examine your radar chart detailing vocabulary richness, sentence complexity, positive/negative polarity, and stylistic punctuation.
3.  **Prompt Evolution (`🧬 Optimize Prompts` page)**:
    *   Input a writing task (e.g., *“Write a speech about workspace productivity.”*)
    *   Select your target evaluation model, number of generations, and population limits.
    *   Click **Run Optimizer** to launch the genetic evolution. Watch the line graph update in real-time as prompts mutate and cross over to score higher matches.
4.  **Imitation Generation (`✍️ Generate Content` page)**:
    *   Input a task and generate text using your top-evolved prompts across multiple models.
    *   Examine outputs ranked by composite style similarity scores with feature-by-feature metrics.
5.  **Analytics & Performance (`🏆 Leaderboard` & `📤 Export` pages)**:
    *   Compare performance metrics across strategies, models, and download the resulting optimized prompt files or style JSONs.
