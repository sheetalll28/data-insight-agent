# Data Insight Agent: Project Review & Finalisation (Nov 2026)

This review covers the full repository and the requirements in `update_instructions.md`.
The goal was to **finalise the existing project**. Only changes that fix incorrect,
broken, misleading or insecure behaviour were made, plus the items that file marks as
critical. It was not used as a mandate for the full 19-phase rebuild it describes. Every
number below was measured by running the code; nothing is estimated.

## 1. What the project contained

| File | Role | State found |
|---|---|---|
| `src/app.py` | Streamlit app: upload/clean, feature engineering, K-Means cohort report, PyGWalker BI | Working UI; several correctness, honesty and security issues (below) |
| `src/pipeline.py` | Cleaning pipeline + Pearson matrix | Working; ID columns treated as data, junk-text coercion never triggered, categorical gaps never filled, O(p²) Python loop |
| `src/outlier_detection.py` | From-scratch IQR + evaluation script | Working; quartiles from a truncated index |
| `src/nn_from_scratch.py` | From-scratch 1-hidden-layer regression NN + gradient check | Backprop correct; 0.01 init left it under-trained; reseeded NumPy's global RNG |
| `src/kmeans_from_scratch.py` | From-scratch K-Means + elbow script | Working; plain random init (poor optima on harder data) |
| `src/dedup_from_scratch.py`, `src/levenshtein.py` | Union-Find entity resolution (research script) | Working; 2 of 5 Levenshtein self-tests had wrong expected values |
| `src/impute_missing_values.py` | NN imputation experiment (research script) | Baseline computed unfairly → false "WIN" |
| `src/corrupt_data.py` | Generates `messy_data.csv` + `ground_truth.json` | Fine |
| `tests/` | Empty | - |
| `README.md` | Docs | Several claims did not match the code |

## 2. Issues found and fixed

| # | Issue | Why it mattered | Fix |
|---|---|---|---|
| 1 | Hard-coded Groq model `llama-3.3-70b-versatile`, **retired by Groq on 16 Aug 2026** | Both AI features were broken | Provider-agnostic BYOK layer (`src/llm_client.py`) for Groq, OpenAI, Gemini, Claude and custom OpenAI-compatible hosts, using the existing `openai` dependency. Default Groq model is `openai/gpt-oss-120b` (Groq's recommended replacement), and any model ID or live model list can be used |
| 2 | K-Means page refused to cluster without `GROQ_API_KEY` | Math depended on an API key | Clustering, profiles and elbow chart always run; the AI report is optional |
| 3 | AI chart plotted `expected_correlation_improvement_pct`, a number **invented by the LLM** | Fabricated metrics presented as results | Removed. Each proposal is now measured on the data (the \|r\| of a z-scored composite vs its best component), and column names the model invented are rejected |
| 4 | NN importance = `mean(|W1|)` | Not a valid importance measure (ignores W2, activations, scale) | Permutation importance on a held-out 20% split |
| 5 | NN trained and judged on all rows; min-max scaling fitted on all rows | No validation; leakage | 80/20 split, scaling fitted on the training rows only, held-out and training R² shown with a warning when R² ≤ 0.05 |
| 6 | NN init `randn × 0.01` | Under-trained after 1000 epochs: held-out R² **0.134** on a synthetic benchmark (linear regression: 0.955) | He initialisation: held-out R² **0.954** with the same epochs, LR and architecture |
| 7 | Identifier columns (e.g. `Patient ID`) clipped, correlated, used as NN features and offered as targets | Spurious "signal" | `detect_identifier_columns()`. IDs are kept in the output but excluded from cleaning and analysis |
| 8 | Comment said junk text like `"UNKNOWN"` becomes NaN, but whole-column `pd.to_numeric` failed, so such columns stayed text and were label-encoded | Numeric columns analysed as categories | Columns that are ≥ 80% numeric are coerced, and the count of junk values is logged |
| 9 | Missing categorical values never filled → code `-1` in the analysis | Silent bias | Mode imputation, logged |
| 10 | IQR quartiles via `sorted[int(n*q)]` | Biased for small n | Linear-interpolated percentiles (NumPy/Excel definition), still implemented from scratch. Identical detection on the bundled data (F1 0.943) |
| 11 | Outliers always clipped silently | Legitimate extremes destroyed without a record | Audit log with counts and bounds; "flag only" option in the UI |
| 12 | Deduplication ran after imputation, silently | Duplicates biased the means; no visibility | Runs first; exact duplicates removed and counted; rows identical except for their ID are reported, not merged |
| 13 | New upload did not clear the previous file's cleaned data; CSV re-parsed on every rerun; malformed CSV crashed the page | Stale or incorrect results | Tracked by `file_id`; parsed once; friendly error |
| 14 | Pearson matrix: Python double loop | Slow on wide data | Vectorised (still from scratch): **7× / 10× / 31×** faster at 500×16 / 10k×50 / 50k×100, max difference vs the original 2.2e-16 |
| 15 | K-Means random init | Poor optima on harder data | K-Means++ seeding, vectorised distances, empty-cluster recovery, validation of `k`. On 10 datasets with 15 clusters (n_init=3), WCSS relative to scikit-learn went from a mean of **4.78×** (worst 12.1×) to **1.07×** (worst 1.30×). Equal on easy blobs |
| 16 | `np.random.seed()` called inside the NN and K-Means | Changed global randomness for the whole app | Private `default_rng(seed)` |
| 17 | `impute_missing_values.py` baseline used the unclipped ages (999, -15 ...) | Printed a false "WIN" (13.17 vs 22.30) | Fair baseline on the same clipped targets: NN 14.36 vs mean 12.83, **the baseline wins**. Documented |
| 17b | `impute_missing_values.py` built its one-hot columns from `list(set(...))` | Results varied from run to run (13.7 to 14.7 RMSE) because Python randomises string hashing | `sorted(set(...))`: reproducible 14.36 |
| 18 | Levenshtein self-test expected values 13 and 7 | 2/5 tests "FAILED" although the algorithm was right | Corrected to 10 and 8 (checked against an independent implementation) |
| 19 | LLM prompts hard-wired to clinical framing ("treatment recommendations") for any CSV | Unsupported domain claims | Generic, evidence-only prompts that cite computed numbers |
| 20 | PyGWalker default privacy level `events` (telemetry + update check) | Data app phoning home; also hung in restricted networks | `set_privacy("offline")` |
| 21 | `streamlit.components.v1.html` and `use_container_width` deprecated (removal date passed) | Would break on a Streamlit upgrade | `st.iframe`, `width="stretch"` |
| 22 | README/sidebar claimed Union-Find dedup and NN imputation run in the app, plus a sigmoid output layer | Inaccurate claims (risky for a marketplace listing) | Docs and sidebar now describe what actually runs; research scripts documented separately |
| 23 | No tests | - | 61 pytest tests (below) |
| 24 | Type repair checked `dtype == 'object'`; pandas 3 stores text as `str` | On pandas ≥ 3 the "UNKNOWN" → NaN repair silently never ran, so numeric columns were analysed as categories | dtype-agnostic check; tests pass on pandas 2.3 and 3.0 |
| 25 | Identifier detection ran before exact dedup | A copied row repeats its ID, hiding the ID column | Detection now runs after dedup |
| 26 | App rendered on Streamlit's light base theme under a forced dark background | Navigation, widget labels and legends were nearly invisible in light-mode browsers | `.streamlit/config.toml` dark theme; UI layer `src/ui.py` (navigation, KPI cards, tabs, validated chart palette) |

## 3. Test suite (`pytest -q` → 61 passed, ~3 s, no keys, no network)

- **Math vs references**: IQR vs `np.percentile` (7 sizes); Pearson vs `np.corrcoef`; backprop vs finite differences (< 1e-6); NN learns a known signal (R² > 0.9) and permutation importance ranks the true driver first; K-Means WCSS equals scikit-learn's inertia (rel. 1e-6, ARI > 0.99); Levenshtein vs an independent implementation.
- **Pipeline**: input not mutated, coercion, mode imputation, ID detection, clip vs flag, exact-only dedup, edge cases (1 row, all-NaN, all-text, constant), bundled dataset fully cleaned.
- **LLM layer (mocked)**: all 5 providers, env-var fallback, readiness messages, JSON mode only where supported, JSON extraction and one corrective retry, 401/404/429/5xx/timeout/connection errors mapped to friendly messages **that never contain the key**, retry and timeout configuration.
- **App (Streamlit AppTest, headless)**: every page renders with no API key; clustering works without a key; held-out R² is shown; AI feature suggestions are measured and invented columns rejected; cohort report uses the configured provider; BI studio renders offline.
- Not automated: the file-upload widget (AppTest cannot drive it) and live provider calls (no real keys used).

## 4. `update_instructions.md` coverage

| Phase | Status | Notes |
|---|---|---|
| 1 Audit & baseline | ✅ | This document |
| 2 Dataset understanding engine | ◐ Partial | ID and type detection, audit log. No semantic or target inference |
| 3 Intelligent cleaning | ◐ Partial | Audit log, clip/flag choice, mode imputation, exact-only dedup, original preserved. No KNN imputation, z-score/robust variants or per-column overrides |
| 4 Statistical discovery | ◐ Partial | Pearson verified and vectorised. No Spearman, Kendall, p-values, FDR or VIF |
| 5 Feature engineering | ◐ Partial | LLM suggestions validated programmatically. No automated polynomial/ratio search |
| 6 NN optimisation | ◐ Partial | Init fix, held-out evaluation, permutation importance, gradient check test. No Adam, mini-batches, classification heads or early stopping |
| 7 Clustering | ◐ Partial | K-Means++, restarts, vectorisation, empty clusters, elbow chart, profiles. No silhouette/DB/CH metrics or alternative algorithms |
| 8 Orchestrator | ✗ Not started | - |
| 9 AutoML benchmarking | ✗ Not started | Would add scikit-learn, XGBoost and others as runtime dependencies |
| 10 Anomaly detection module | ◐ Minimal | IQR flags in the audit log only |
| 11 Multi-LLM BYOK | ✅ Core done | 5 providers, settings page, test connection, live model list, env fallback, timeouts and retries, key scrubbing, token usage as reported. No per-provider native SDKs, cost estimates or encrypted key persistence (deliberate: keys stay in memory) |
| 12 Insight generation | ◐ Partial | Evidence-only prompts for the two existing AI features. No full executive report |
| 13 Conversational analyst | ✗ Not started | Requires a safe query engine |
| 14 BI studio | ◐ Partial | PyGWalker offline mode, sampled charts, elbow chart |
| 15 Architecture | ◐ Partial | Separate LLM module, caching of expensive steps, tests. App remains a single-file Streamlit script |
| 16 UI/UX | ◐ Partial | Existing theme kept; AI Settings page and status in the sidebar |
| 17 Performance | ◐ Partial | Measured wins in §2 (Pearson, K-Means, NN). No large-file profiling |
| 18 Testing | ✅ Core done | §3 |
| 19 Deliverables | ◐ | Code, tests, README, `.env.example`, this report |

## 5. Repository hygiene notes

- `.env` is git-ignored and **has never been committed** (checked with `git log --all -- .env`).
- `image.png` (an untracked screenshot of an email thread containing personal email addresses) sits in the repo root. Do not commit it.
- `update_instructions.md` is untracked. Decide whether it belongs in the public repo.
- `env/` is a macOS virtual environment and is git-ignored; recreate it on other machines with `requirements.txt`.
- `faker` is in `requirements.txt` but is not imported anywhere.
- These changes are **not committed**. Review them with `git diff` and commit when ready.
