import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import io
from html import escape
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import (run_cleaning_pipeline, calculate_correlation_matrix, detect_identifier_columns,
                      read_tabular_file, SUPPORTED_EXTENSIONS)
from nn_from_scratch import (NeuralNetwork, train_test_split_indices, minmax_fit, minmax_transform,
                             r2_score, permutation_importance)
from kmeans_from_scratch import KMeansFromScratch
from llm_client import PROVIDERS, LLMConfig, LLMError, chat, chat_json, list_models, test_connection
import ui

# --- SETUP AND PREMIUM STYLING ---
st.set_page_config(page_title="Data Insight Agent", page_icon=":material/insights:", layout="wide", initial_sidebar_state="expanded")
ui.inject_theme()
ui.register_plotly_theme()

PAGES = {
    "upload": ":material/upload_file: Upload & Clean",
    "features": ":material/query_stats: Feature Discovery",
    "cohorts": ":material/hub: Cohort Intelligence",
    "bi": ":material/dashboard: BI Studio",
    "settings": ":material/tune: AI Settings",
}

# --- STATE MANAGEMENT ---
for _key, _default in {
    'raw_df': None, 'clean_df': None, 'clean_report': None, 'file_id': None,
    'llm_config': LLMConfig(), 'llm_models': [], 'llm_usage': [], 'cluster_result': None,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def analysis_frame(clean_df):
    """Numeric view for the math engines: identifiers dropped, categories label-encoded."""
    ids = detect_identifier_columns(clean_df)
    df = clean_df.drop(columns=ids).copy()
    categorical = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]
            categorical.append(col)
    return df, ids, categorical


@st.cache_data(show_spinner=False)
def cached_correlation(df):
    return calculate_correlation_matrix(df)


@st.cache_data(show_spinner=False)
def train_and_explain(X, y, feature_cols, epochs=1000, seed=42):
    """
    Train the from-scratch NN on 80% of rows, evaluate on the held-out 20%.
    Scaling is fitted on the training split only (no leakage).
    Importance = permutation importance on the held-out split.
    """
    tr, te = train_test_split_indices(len(X), test_size=0.2, seed=seed)
    mins, maxs = minmax_fit(X[tr])
    Xtr, Xte = minmax_transform(X[tr], mins, maxs), minmax_transform(X[te], mins, maxs)
    y_min, y_max = y[tr].min(), y[tr].max()
    ytr = ((y[tr] - y_min) / (y_max - y_min + 1e-8)).reshape(-1, 1)
    yte = ((y[te] - y_min) / (y_max - y_min + 1e-8)).reshape(-1, 1)

    nn = NeuralNetwork(input_size=X.shape[1], hidden_size=8, output_size=1, learning_rate=0.1, seed=seed)
    nn.fit(Xtr, ytr, epochs=epochs)
    test_r2 = r2_score(yte, nn.predict(Xte))
    train_r2 = r2_score(ytr, nn.predict(Xtr))
    imp_mean, imp_std = permutation_importance(nn.predict, Xte, yte, n_repeats=10, seed=seed)
    imp = pd.DataFrame({'Feature': feature_cols, 'Importance': imp_mean, 'Std': imp_std})
    return imp, float(train_r2), float(test_r2), int(len(tr)), int(len(te))


def zscore_composite(df, cols):
    """Equal-weight average of z-scored features: a transparent way to test an AI-suggested composite."""
    parts = []
    for c in cols:
        s = df[c].astype(float)
        sd = s.std()
        if sd > 0:
            parts.append((s - s.mean()) / sd)
    return pd.concat(parts, axis=1).mean(axis=1) if parts else None


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def column_overview(raw):
    """One row per column: detected type, missing share, distinct values."""
    ids = set(detect_identifier_columns(raw.drop_duplicates()))
    rows = []
    for col in raw.columns:
        s = raw[col]
        if col in ids:
            kind = "Identifier"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "Numeric"
        else:
            non_empty = s.dropna()
            share = pd.to_numeric(non_empty, errors='coerce').notna().mean() if len(non_empty) else 0
            kind = "Numeric (needs repair)" if share >= 0.8 else "Categorical"
        rows.append({"Column": str(col), "Type": kind, "Missing": float(s.isna().mean()),
                     "Distinct values": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def record_usage(result):
    st.session_state['llm_usage'].append({k: result.get(k) for k in
                                          ('provider', 'model', 'seconds', 'input_tokens', 'output_tokens')})


def ai_unavailable_notice():
    cfg = st.session_state['llm_config']
    st.info(f"AI narrative is optional. {cfg.missing()} The mathematical results above work without it.")


def need_clean_data():
    st.warning("No cleaned dataset yet. Open **Upload & Clean**, upload a CSV and run the pipeline first.")


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    ui.sidebar_brand()
    ui.sidebar_label("Workspace")
    menu_label = st.radio("Navigation", list(PAGES.values()), label_visibility="collapsed")
    menu = {v: k for k, v in PAGES.items()}[menu_label]

    ui.sidebar_label("Progress")
    raw_ok = st.session_state['raw_df'] is not None
    clean_ok = st.session_state['clean_df'] is not None
    cfg = st.session_state['llm_config']
    ui.step("Dataset uploaded", raw_ok)
    ui.step("Pipeline run", clean_ok)
    ui.step("AI connected (optional)", cfg.is_ready())
    if clean_ok:
        c = st.session_state['clean_df']
        st.caption(f"{c.shape[0]:,} rows · {c.shape[1]} columns ready")
    if cfg.is_ready():
        st.caption(f"AI: {cfg.provider} · {cfg.resolved_model()}")

    ui.sidebar_label("Engines")
    with st.expander("Built from scratch (NumPy)"):
        st.markdown("""
        - IQR Outlier Engine
        - Pearson Correlation Engine
        - Neural Network + Permutation Importance
        - K-Means++ Clustering
        """)
        st.caption("Union-Find entity resolution and NN imputation live in standalone "
                   "research scripts in `src/` (see README).")

# --- PAGE ROUTING ---
if menu == "upload":
    ui.page_header("Step 1 · Data pipeline", "Upload & Auto-Clean",
                   "Upload any CSV or TSV file. Our from-scratch engines repair types, find identifiers, remove exact "
                   "duplicates, handle outliers and fill gaps, and every change is logged.")

    # 1. FILE UPLOADER
    uploaded_file = st.file_uploader("Upload a CSV or TSV file", type=list(SUPPORTED_EXTENSIONS),
                                     help="Comma-separated (.csv) or tab-separated (.tsv). "
                                          "Try data/demo/retail_customers_demo.csv")

    if uploaded_file is not None and uploaded_file.file_id != st.session_state['file_id']:
        # New file: read it once and invalidate results from any previous file
        try:
            raw = read_tabular_file(uploaded_file, uploaded_file.name)
            st.session_state['raw_df'] = raw
        except Exception as e:
            st.session_state['raw_df'] = None
            st.error(f"Could not read {uploaded_file.name}: {e}")
        st.session_state['file_id'] = uploaded_file.file_id
        st.session_state['clean_df'] = None
        st.session_state['clean_report'] = None
        st.session_state['cluster_result'] = None

    if st.session_state['raw_df'] is None:
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.card("<p>Drop in a CSV or TSV file. Nothing leaves your machine during cleaning.</p>", "Upload", "01")
        with c2:
            ui.card("<p>One click runs the IQR, imputation and deduplication engines, with a full audit log.</p>",
                    "Clean", "02")
        with c3:
            ui.card("<p>Find drivers, clusters and patterns, then let your chosen AI explain them.</p>",
                    "Discover", "03")
    else:
        raw = st.session_state['raw_df']
        overview = column_overview(raw)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{raw.shape[0]:,}")
        k2.metric("Columns", f"{raw.shape[1]}")
        k3.metric("Empty cells", f"{raw.isna().mean().mean() * 100:.1f}%")
        k4.metric("Exact duplicate rows", f"{int(raw.duplicated().sum()):,}")

        ui.section("Column overview", "Detected type, missing values and cardinality of each column.")
        st.dataframe(
            overview, width="stretch", hide_index=True,
            column_config={"Missing": st.column_config.ProgressColumn(
                "Missing", format="percent", min_value=0.0, max_value=1.0)})

        ui.section("Run the pipeline")
        o1, o2 = st.columns([3, 1])
        with o1:
            clip = st.checkbox("Clip IQR outliers to the Tukey bounds (untick to only flag them)", value=True,
                               help="Extreme values are not always errors. Unticking keeps them and just reports them.")
        with o2:
            run = st.button("Run Auto-Clean Pipeline", icon=":material/play_arrow:", type="primary", width="stretch")
        if run:
            with st.spinner("Running IQR Outlier Detection, Imputation, and Deduplication..."):
                cleaned, report = run_cleaning_pipeline(raw, clip_outliers=clip, return_report=True)
                st.session_state['clean_df'] = cleaned
                st.session_state['clean_report'] = report
                st.session_state['cluster_result'] = None
                st.rerun()

    if st.session_state['clean_df'] is not None:
        report = st.session_state['clean_report'] or {}
        n_out = sum(v['count'] for v in report.get('outliers', {}).values())
        n_imp = sum(v['count'] for v in report.get('imputed', {}).values())
        st.success(f"Pipeline complete: {report.get('rows_out', 0):,} clean rows, {n_out:,} outlier cells handled, "
                   f"{n_imp:,} gaps filled. Continue to **Feature Discovery** in the sidebar.")

        tab_log, tab_clean, tab_raw = st.tabs([":material/fact_check: Audit log", ":material/table_view: Cleaned data", ":material/description: Raw data"])
        with tab_log:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows in → out", f"{report.get('rows_in', 0):,} → {report.get('rows_out', 0):,}")
            c2.metric("Exact duplicates removed", report.get('exact_duplicates_removed', 0))
            c3.metric("Outlier cells", n_out)
            c4.metric("Cells imputed", n_imp)
            notes = []
            if report.get('identifier_columns'):
                notes.append("<b>Identifier columns</b> kept, but excluded from cleaning and analysis: " +
                             ", ".join(f"<code>{escape(str(c))}</code>" for c in report['identifier_columns']))
            if report.get('coerced_to_numeric'):
                notes.append("<b>Text columns converted to numbers</b> (non-numeric entries became missing): " +
                             ", ".join(f"<code>{escape(str(c))}</code> ({n})" for c, n in report['coerced_to_numeric'].items()))
            if report.get('near_duplicates_ignoring_ids'):
                notes.append(f"<b>{report['near_duplicates_ignoring_ids']} rows</b> match another row except for "
                             "their identifier. They were <b>kept</b>: identical-looking rows are not assumed to be the "
                             "same entity.")
            if notes:
                st.markdown("")
                ui.card("".join(f"<p>{n}</p>" for n in notes), "What the pipeline noticed")
            l1, l2 = st.columns(2)
            with l1:
                ui.section("IQR outliers", "Tukey bounds per column (Q1 − 1.5·IQR, Q3 + 1.5·IQR).")
                if report.get('outliers'):
                    out_df = pd.DataFrame(report['outliers']).T.rename_axis("Column").reset_index()
                    out_df[["lower", "upper"]] = out_df[["lower", "upper"]].astype(float).round(2)
                    out_df.columns = [c.title() if c != "Column" else c for c in out_df.columns]
                    st.dataframe(out_df, width="stretch", hide_index=True)
                else:
                    st.caption("No outliers found.")
            with l2:
                ui.section("Missing values filled", "Numeric columns use the mean, categorical the mode.")
                if report.get('imputed'):
                    imp_df = pd.DataFrame(report['imputed']).T.rename_axis("Column").reset_index()
                    imp_df.columns = [c.title() if c != "Column" else c for c in imp_df.columns]
                    st.dataframe(imp_df, width="stretch", hide_index=True)
                else:
                    st.caption("No missing values.")
        with tab_clean:
            st.dataframe(st.session_state['clean_df'].head(200), width="stretch")
            # 3. DOWNLOAD BUTTON
            csv_buffer = io.StringIO()
            st.session_state['clean_df'].to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download cleaned CSV", icon=":material/download:",
                data=csv_buffer.getvalue(),
                file_name="cleaned_insight_data.csv",
                mime="text/csv",
            )
        with tab_raw:
            st.dataframe(st.session_state['raw_df'].head(200), width="stretch")

elif menu == "features":
    ui.page_header("Step 2 · Feature discovery", "Feature Engineering & Analytics",
                   "See how every feature relates to your target, which ones a neural network actually relies "
                   "on, and test AI-proposed composite features against the data.")

    if st.session_state['clean_df'] is None:
        need_clean_data()
    else:
        df, id_cols, cat_cols = analysis_frame(st.session_state['clean_df'])
        numeric_cols = df.columns.tolist()

        if len(numeric_cols) < 2:
            st.error("At least two usable (non-identifier) columns are needed for feature engineering.")
        else:
            t1, t2 = st.columns([2, 3])
            with t1:
                # Dynamic Target Selection
                target_col = st.selectbox("Target variable", numeric_cols, index=len(numeric_cols) - 1,
                                          help="Defaults to the last column, where targets usually live.")
            with t2:
                st.markdown("")
                note = (" Categorical: label codes are treated as numbers, so results show only a rough trend."
                        if target_col in cat_cols else "")
                st.caption(f"{len(df):,} rows · {len(numeric_cols) - 1} candidate features"
                           + (f" · identifiers excluded: {', '.join(map(str, id_cols))}" if id_cols else "") + "." + note)

            corr_df = cached_correlation(df)
            correlations = corr_df[target_col].drop(target_col).sort_values(key=abs, ascending=True)

            tab_corr, tab_nn, tab_ai = st.tabs([":material/scatter_plot: Correlations", ":material/neurology: Neural importance", ":material/auto_awesome: AI feature ideas"])

            with tab_corr:
                strongest = correlations.abs().idxmax() if len(correlations) else None
                m1, m2, m3 = st.columns(3)
                m1.metric("Strongest relationship", str(strongest) if strongest is not None else "-")
                m2.metric("Pearson r", f"{correlations[strongest]:+.2f}" if strongest is not None else "-")
                m3.metric("Features with |r| ≥ 0.3", int((correlations.abs() >= 0.3).sum()))

                fig_bar = go.Figure(go.Bar(
                    x=correlations.values, y=[str(i) for i in correlations.index], orientation='h',
                    marker=dict(color=[ui.POSITIVE if v >= 0 else ui.NEGATIVE for v in correlations.values],
                                cornerradius=4),
                    text=[f"{v:+.2f}" for v in correlations.values], textposition="outside",
                    textfont=dict(color=ui.TEXT_MUTED, size=11), cliponaxis=False,
                    hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"))
                fig_bar.update_layout(
                    title=f"Pearson correlation with {target_col}",
                    height=max(260, 34 * len(correlations) + 90), bargap=0.35,
                    xaxis=dict(range=[-1.15, 1.15], title="Correlation coefficient (r)", zeroline=True))
                ui.chart(fig_bar)
                st.caption("Blue = moves with the target, red = moves against it. Correlation is not causation; "
                           "|r| below ~0.1 is usually indistinguishable from noise.")

                if len(numeric_cols) <= 30:
                    order = corr_df.columns.tolist()
                    fig_hm = go.Figure(go.Heatmap(
                        z=corr_df.values, x=[str(c) for c in order], y=[str(c) for c in order],
                        colorscale=ui.DIVERGING, zmin=-1, zmax=1, xgap=2, ygap=2,
                        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
                        colorbar=dict(title=dict(text="r", font=dict(color=ui.TEXT_MUTED)),
                                      tickvals=[-1, 0, 1], ticktext=["−1", "0", "+1"])))
                    fig_hm.update_layout(title="Correlation matrix (all features)", margin=dict(l=10, r=70, t=48, b=10),
                                         height=min(720, 120 + 40 * len(order)),
                                         yaxis=dict(autorange="reversed", showgrid=False),
                                         xaxis=dict(showgrid=False, tickangle=-35))
                    ui.chart(fig_hm)

                # Scatter Plots Grid (sampled so large files stay responsive)
                ui.section("Strongest relationships up close",
                           f"The six features most correlated with {escape(str(target_col))} (up to 3,000 sampled rows; "
                           "box plots per class when the target has 10 or fewer values).")
                plot_df = df.sample(n=min(len(df), 3000), random_state=42)
                # Few distinct target values (e.g. a 0/1 flag): compare each feature's
                # distribution per class instead of a two-stripe scatter.
                few_classes = df[target_col].nunique() <= 10
                if few_classes:
                    plot_df = plot_df.assign(**{target_col: st.session_state['clean_df'].loc[plot_df.index, target_col].astype(str)})
                cols = st.columns(3)
                for idx, feature in enumerate(correlations.index[::-1][:6]):
                    if few_classes:
                        fig_scatter = px.box(plot_df, x=target_col, y=feature, points=False,
                                             color_discrete_sequence=[ui.PRIMARY_SERIES])
                        fig_scatter.update_xaxes(type="category")
                    else:
                        fig_scatter = px.scatter(plot_df, x=feature, y=target_col, opacity=0.55,
                                                 color_discrete_sequence=[ui.PRIMARY_SERIES])
                        fig_scatter.update_traces(marker=dict(size=6))
                    fig_scatter.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10),
                                              title=f"{feature}  ·  r = {correlations[feature]:+.2f}")
                    ui.chart(fig_scatter, cols[idx % 3])

            with tab_nn:
                st.markdown(f"Our from-scratch network trains on 80% of rows to predict **{target_col}**, is scored "
                            "on the held-out 20%, and measures **permutation importance**: how much the held-out "
                            "error grows when each feature is shuffled.")
                with st.spinner("Training Neural Network and measuring permutation importance..."):
                    # Cap at the 15 features most correlated with the target to keep the UI responsive
                    feature_cols = correlations.abs().sort_values(ascending=False).index[:15].tolist()
                    X = df[feature_cols].to_numpy(dtype=float)
                    y = df[target_col].to_numpy(dtype=float)
                    if len(X) < 10:
                        st.warning("Too few rows to train and evaluate a model honestly (need at least 10).")
                    else:
                        imp_df, train_r2, test_r2, n_tr, n_te = train_and_explain(X, y, feature_cols)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Held-out R²", f"{test_r2:.3f}")
                        m2.metric("Training R²", f"{train_r2:.3f}")
                        m3.metric("Train / test rows", f"{n_tr:,} / {n_te:,}")
                        if test_r2 <= 0.05:
                            st.warning("The network does not predict this target better than its average on unseen "
                                       "rows, so the importances below mostly reflect noise.")
                        elif test_r2 >= 0.5:
                            st.markdown(ui.badge(f"Strong model: explains {test_r2:.0%} of the variation in unseen rows",
                                                 "good"), unsafe_allow_html=True)
                        else:
                            st.markdown(ui.badge(f"Moderate model: explains {test_r2:.0%} of the variation in unseen rows",
                                                 "warning"), unsafe_allow_html=True)
                        imp_df = imp_df.sort_values(by='Importance', ascending=True)
                        fig2 = go.Figure(go.Bar(
                            x=imp_df['Importance'], y=imp_df['Feature'].astype(str), orientation='h',
                            marker=dict(color=ui.PRIMARY_SERIES, cornerradius=4),
                            error_x=dict(type="data", array=imp_df['Std'], color=ui.TEXT_FAINT, thickness=1.5, width=3),
                            hovertemplate="%{y}<br>Δ MSE = %{x:.4f}<extra></extra>"))
                        fig2.update_layout(
                            title=f"Permutation importance on held-out rows (target: {target_col})",
                            height=max(260, 34 * len(imp_df) + 90), bargap=0.35,
                            xaxis=dict(title="Increase in held-out error when shuffled (scaled MSE)"))
                        ui.chart(fig2)
                        st.caption("Bars show the mean over 10 shuffles; whiskers show ±1 standard deviation.")

            with tab_ai:
                st.markdown("Your selected AI model reads the correlations and proposes composite features. "
                            "Each proposal is then **tested on your data**. The AI never supplies the numbers.")
                cfg = st.session_state['llm_config']
                if not cfg.is_ready():
                    ai_unavailable_notice()
                elif st.button("Generate AI feature recommendations", icon=":material/auto_awesome:", type="primary"):
                    with st.spinner(f"Asking {cfg.provider} ({cfg.resolved_model()})..."):
                        prompt = f"""
Dataset: {len(df):,} rows. Target variable: "{target_col}".
Pearson correlation of each feature with the target (computed exactly; label-encoded columns: {cat_cols or 'none'}):
{correlations.sort_values(key=abs, ascending=False).round(4).to_string()}

Suggest up to 3 composite features, each combining 2-4 of the EXACT feature names above,
that might predict "{target_col}" better than any single feature. Do not invent numbers;
the application will measure each proposal itself. If every |r| is tiny, say so in the reasoning.

Return only a JSON object of this shape:
{{"recommendations": [{{"feature_name": "...", "raw_features_combined": ["...", "..."], "reasoning": "..."}}]}}
"""
                        try:
                            result_json, meta = chat_json(
                                cfg, "You are a careful data scientist. You only output valid JSON.", prompt)
                            record_usage(meta)
                            recs = result_json.get("recommendations", [])
                            rows = []
                            best_single = float(correlations.abs().max()) if len(correlations) else 0.0
                            for r in recs if isinstance(recs, list) else []:
                                used = [f for f in r.get('raw_features_combined', []) if f in df.columns and f != target_col]
                                rejected = [f for f in r.get('raw_features_combined', []) if f not in used]
                                comp = zscore_composite(df, used) if len(used) >= 2 else None
                                measured = abs(pearson(comp, df[target_col])) if comp is not None else None
                                best_part = max((abs(correlations.get(f, 0.0)) for f in used), default=0.0)
                                rows.append({
                                    "Proposed feature": r.get('feature_name', '?'),
                                    "Combines": ", ".join(used) or "-",
                                    "Measured |r| of composite": None if measured is None else round(measured, 4),
                                    "Best single component |r|": round(best_part, 4),
                                    "Beats its components?": None if measured is None else ("Yes" if measured > best_part else "No"),
                                    "Note": ("Unknown columns ignored: " + ", ".join(map(str, rejected))) if rejected else "",
                                    "_reasoning": r.get('reasoning') or r.get('mathematical_reasoning', ''),
                                })
                            if rows:
                                st.success(f"Measured {len(rows)} AI proposals on your data "
                                           f"({meta['provider']} · {meta['model']} · {meta['seconds']}s).")
                                cards = st.columns(len(rows))
                                for col, r in zip(cards, rows):
                                    verdict = r["Beats its components?"]
                                    b = (ui.badge("Beats its best component", "good") if verdict == "Yes" else
                                         ui.badge("No gain over its best component", "neutral") if verdict == "No" else
                                         ui.badge("Could not be measured", "warning"))
                                    measured_txt = ("-" if r["Measured |r| of composite"] is None
                                                    else f'{r["Measured |r| of composite"]:.3f}')
                                    with col:
                                        ui.card(
                                            f'<p>{b}</p><p><span class="big">{measured_txt}</span> measured |r| '
                                            f'(best part {r["Best single component |r|"]:.3f})</p>'
                                            f'<p><b>Combines:</b> {escape(r["Combines"])}</p>'
                                            f'<p><i>AI hypothesis:</i> {escape(str(r["_reasoning"]))}</p>'
                                            + (f'<p>{ui.badge(r["Note"], "warning")}</p>' if r["Note"] else ""),
                                            title=str(r["Proposed feature"]))
                                ui.section("Measurement table")
                                st.dataframe(pd.DataFrame(rows).drop(columns=['_reasoning']), width="stretch",
                                             hide_index=True)
                                st.caption(f"Composites are an equal-weight average of z-scored components. "
                                           f"Strongest single feature overall: |r| = {best_single:.4f}. "
                                           "Measured on all rows, so treat gains as exploratory, not validated.")
                            else:
                                st.warning("The AI returned no usable recommendations.")
                        except LLMError as e:
                            st.error(str(e))

elif menu == "cohorts":
    ui.page_header("Step 3 · Unsupervised learning", "Cohort Intelligence",
                   "Our from-scratch K-Means++ engine groups similar rows. Profiles are computed from the data; "
                   "a configured AI can then describe them in plain English.")

    if st.session_state['clean_df'] is None:
        need_clean_data()
    else:
        df, id_cols, cat_cols = analysis_frame(st.session_state['clean_df'])
        numeric_cols = df.columns.tolist()
        default_cols = [c for c in numeric_cols if c not in cat_cols][:4] or numeric_cols[:min(4, len(numeric_cols))]

        s1, s2, s3 = st.columns([4, 2, 1.4])
        with s1:
            cluster_cols = st.multiselect("Features for clustering", numeric_cols, default=default_cols,
                                          help="Numeric measures work best; categorical columns are label-encoded.")
        with s2:
            k = st.slider("Number of clusters (K)", min_value=2, max_value=ui.MAX_SERIES, value=4,
                          help="Use the elbow chart to judge a sensible K.")
        with s3:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            run = st.button("Run Clustering", icon=":material/play_arrow:", type="primary", width="stretch")

        if run:
            if len(cluster_cols) < 2:
                st.error("Please select at least 2 features to cluster.")
            elif len(df) < k:
                st.error(f"K = {k} needs at least {k} rows.")
            else:
                with st.spinner("Running Mathematical Clustering Engine..."):
                    X = df[cluster_cols].to_numpy(dtype=float)
                    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)

                    kmeans = KMeansFromScratch(k=k, n_init=5)
                    labels = kmeans.fit(X_norm)
                    elbow = []
                    for kk in range(2, min(10, len(df)) + 1):
                        m = KMeansFromScratch(k=kk, n_init=3)
                        m.fit(X_norm)
                        elbow.append({'K': kk, 'WCSS': m.wcss})
                st.session_state['cluster_result'] = {
                    "cols": list(cluster_cols), "k": k, "labels": labels, "elbow": elbow,
                    "wcss": kmeans.wcss, "report": None}

        res = st.session_state['cluster_result']
        if res and res["cols"] == list(cluster_cols) and res["k"] == k:
            labels, elbow = res["labels"], res["elbow"]
            names = [f"Cluster {i + 1}" for i in range(k)]
            view = st.session_state['clean_df'].loc[df.index, cluster_cols].copy()
            view['Cluster'] = [names[i] for i in labels]
            numeric_view = [c for c in cluster_cols if c not in cat_cols]
            sizes = view['Cluster'].value_counts().reindex(names, fill_value=0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clusters", k)
            m2.metric("Rows clustered", f"{len(view):,}")
            m3.metric("Largest cluster", f"{sizes.max() / sizes.sum():.0%}")
            m4.metric("Smallest cluster", f"{sizes.min() / sizes.sum():.0%}")

            ui.section("1. Pattern discovery", "Each cluster has its own colour and marker shape.")
            p1, p2 = st.columns([3, 2])
            x_col, y_col = cluster_cols[0], cluster_cols[1]
            z_col = cluster_cols[2] if len(cluster_cols) > 2 else cluster_cols[0]
            plot_df = df.assign(Cluster=[names[i] for i in labels]).sample(n=min(len(df), 5000), random_state=42)
            with p1:
                fig = px.scatter_3d(
                    plot_df, x=x_col, y=y_col, z=z_col, color='Cluster', symbol='Cluster',
                    category_orders={"Cluster": names}, color_discrete_sequence=ui.SERIES,
                    symbol_sequence=ui.SYMBOLS, title="Clusters in feature space", opacity=0.85)
                fig.update_traces(marker=dict(size=4, line=dict(width=0)))
                fig.update_layout(height=520, margin=dict(l=0, r=0, b=0, t=48),
                                  legend=dict(orientation="h", y=-0.02, title_text=""))
                ui.chart(fig)
            with p2:
                el = pd.DataFrame(elbow)
                fig_elbow = go.Figure(go.Scatter(
                    x=el['K'], y=el['WCSS'], mode="lines+markers",
                    line=dict(color=ui.PRIMARY_SERIES, width=2),
                    marker=dict(size=8, color=[ui.ACCENT if kk == k else ui.PRIMARY_SERIES for kk in el['K']],
                                line=dict(width=2, color=ui.BG)),
                    hovertemplate="K = %{x}<br>WCSS = %{y:.2f}<extra></extra>"))
                fig_elbow.update_layout(title="Elbow method", height=520, margin=dict(l=10, r=24, t=48, b=10),
                                        xaxis=dict(title="K (number of clusters)", dtick=1,
                                                   range=[el['K'].min() - 0.4, el['K'].max() + 0.4]),
                                        yaxis=dict(title="Within-cluster sum of squares"))
                ui.chart(fig_elbow)

            # Cluster profiles (computed)
            profile = view.groupby('Cluster')[numeric_view].mean().round(2).reindex(names) if numeric_view \
                else pd.DataFrame(index=names)
            profile.insert(0, 'Size', sizes)
            profile.insert(1, 'Share %', (sizes / sizes.sum() * 100).round(1))
            for c in [c for c in cluster_cols if c in cat_cols]:
                profile[f"{c} (most common)"] = view.groupby('Cluster')[c].agg(lambda s: s.mode().iloc[0]).reindex(names)

            st.markdown("### 2. Cluster Profiles")
            st.caption("What makes each cluster different from the average row (in standard deviations, σ).")
            overall_mean = view[numeric_view].mean() if numeric_view else None
            overall_std = view[numeric_view].std().replace(0, np.nan) if numeric_view else None
            per_row = 4
            for start in range(0, k, per_row):
                cols = st.columns(per_row)
                for i in range(start, min(start + per_row, k)):
                    name = names[i]
                    lines = []
                    if numeric_view:
                        z = ((profile.loc[name, numeric_view] - overall_mean) / overall_std).dropna()
                        for feat in z.abs().sort_values(ascending=False).index[:3]:
                            val = z[feat]
                            arrow = "▲" if val > 0 else "▼"
                            lines.append(f'<div class="dia-feature"><span>{arrow} {escape(str(feat))}<br>'
                                         f'<small>avg {profile.loc[name, feat]:,.2f}</small></span>'
                                         f'<b>{val:+.1f}σ</b></div>')
                    for c in [c for c in cluster_cols if c in cat_cols]:
                        lines.append(f'<div class="dia-feature"><span>{escape(str(c))}</span>'
                                     f'<b>{escape(str(profile.loc[name, f"{c} (most common)"]))}</b></div>')
                    with cols[i - start]:
                        ui.card(f'<p>{ui.series_dot(i)}<span class="big">{int(sizes[name]):,}</span> rows · '
                                f'{profile.loc[name, "Share %"]:.1f}%</p>' + "".join(lines), title=name)
            with st.expander("Full profile table"):
                st.dataframe(profile, width="stretch")

            ui.section("3. AI pattern analysis", "Written from the computed profile table only.")
            cfg = st.session_state['llm_config']
            if not cfg.is_ready():
                ai_unavailable_notice()
            else:
                if res.get("report") is None:
                    with st.spinner(f"Asking {cfg.provider} ({cfg.resolved_model()})..."):
                        prompt = f"""
A dataset of {len(df):,} rows was clustered into {k} groups with K-Means on min-max-scaled features.
Computed cluster profiles (Size = rows; numeric columns = cluster means in original units;
"(most common)" columns = most frequent category):

{profile.to_string()}

Overall means for comparison:
{view[numeric_view].mean().round(2).to_string() if numeric_view else 'n/a'}

Write a brief, professional report on these {k} cohorts in Markdown:
- What distinguishes each cohort, citing only the numbers above.
- Say plainly where cohorts barely differ.
- Suggest what a domain expert might investigate next.
Do not invent statistics, causes, or diagnoses that the numbers do not show.
"""
                        try:
                            result = chat(cfg, "You are an expert data scientist who only reports what the data shows.", prompt)
                            record_usage(result)
                            res["report"] = result
                        except LLMError as e:
                            st.error(str(e))
                if res.get("report"):
                    result = res["report"]
                    with st.container(border=True):
                        st.markdown(result["text"])
                    st.caption(f"Generated by {result['provider']} · {result['model']} in {result['seconds']}s. "
                               "AI interpretation: verify it against the computed profiles above.")
        elif res:
            st.info("Settings changed. Click **Run Clustering** to update the results.")
        else:
            st.caption("Choose features and K, then click **Run Clustering**.")

elif menu == "bi":
    ui.page_header("Step 4 · Explore", "Business Intelligence Studio",
                   "Tableau/Power BI-style drag-and-drop exploration of your cleaned data. "
                   "Runs offline: no telemetry leaves your machine.")

    if st.session_state['clean_df'] is None:
        need_clean_data()
    else:
        import pygwalker as pyg
        # Offline mode: no usage telemetry or update checks leave the machine
        # (PyGWalker's default privacy level is "events").
        pyg.GlobalVarManager.set_privacy("offline")

        with st.spinner("Rendering Business Intelligence Studio..."):
            # Generate pure HTML payload instead of hacking Streamlit's server
            html = pyg.to_html(st.session_state['clean_df'], dark="dark")

            # Embed the HTML directly into Streamlit (st.iframe replaces the
            # deprecated streamlit.components.v1.html)
            st.iframe(html, height=900)

elif menu == "settings":
    ui.page_header("Configuration", "AI Settings",
                   "Bring your own key for any provider. Everything else in the app works without one. Only "
                   "aggregated statistics (correlations, cluster averages) are sent, never your raw rows.")

    cfg = st.session_state['llm_config']
    key_source = "This session" if cfg.api_key else (".env file" if cfg.resolved_key() else "None")
    ui.status_grid([
        ("Status", ui.badge("Ready", "good") if cfg.is_ready() else ui.badge("Not configured", "neutral")),
        ("Provider", escape(cfg.provider)),
        ("Model", f"<code>{escape(cfg.resolved_model() or '-')}</code>"),
        ("Key source", escape(key_source)),
    ])
    if not cfg.is_ready():
        st.info(cfg.missing())

    left, right = st.columns([3, 2])
    with left:
        ui.section("Connection")
        provider = st.selectbox("AI Provider", list(PROVIDERS), index=list(PROVIDERS).index(cfg.provider))
        info = PROVIDERS[provider]
        env_key_present = bool(os.environ.get(info['env_var']))

        with st.form("ai_settings"):
            api_key = st.text_input(
                "API Key", type="password", value="",
                placeholder=("Leave blank to use " + info['env_var'] + " from .env") if env_key_present else "Paste your key",
                help=f"Get a key at {info['key_help']}. Kept only in this browser session's memory; never saved or logged.")
            base_url = ""
            if provider == "Custom (OpenAI-compatible)":
                base_url = st.text_input("Base URL", value=cfg.base_url if cfg.provider == provider else "",
                                         placeholder="https://openrouter.ai/api/v1")
            models = st.session_state['llm_models'] if cfg.provider == provider else []
            current_model = cfg.model if cfg.provider == provider else ""
            if models:
                picked = st.selectbox("Model", ["(type below)"] + models,
                                      index=(models.index(current_model) + 1) if current_model in models else 0)
            else:
                picked = "(type below)"
            typed_model = st.text_input("Model ID" + (" (used when '(type below)' is selected)" if models else ""),
                                        value="" if current_model in models else current_model,
                                        placeholder=info['default_model'] or "e.g. a model from 'Fetch models'")
            timeout = st.number_input("Request timeout (seconds)", 10, 300, int(cfg.timeout))
            saved = st.form_submit_button("Save configuration", icon=":material/save:", type="primary")

        if saved:
            model = picked if picked != "(type below)" else typed_model
            keep_key = cfg.api_key if (cfg.provider == provider and not api_key) else api_key
            st.session_state['llm_config'] = LLMConfig(provider=provider, api_key=keep_key, model=model.strip(),
                                                       base_url=base_url.strip(), timeout=float(timeout))
            if provider != cfg.provider:
                st.session_state['llm_models'] = []
            st.rerun()

        cfg = st.session_state['llm_config']
        b1, b2, b3 = st.columns(3)
        if b1.button("Fetch models", icon=":material/refresh:", width="stretch"):
            try:
                st.session_state['llm_models'] = list_models(cfg)
                st.success(f"Found {len(st.session_state['llm_models'])} models. Pick one above and save.")
                st.rerun()
            except LLMError as e:
                st.error(str(e))
        if b2.button("Test connection", icon=":material/wifi_tethering:", width="stretch"):
            try:
                r = test_connection(cfg)
                record_usage(r)
                st.success(f"Connected to {r['provider']} · {r['model']} in {r['seconds']}s.")
            except LLMError as e:
                st.error(str(e))
        if b3.button("Clear", icon=":material/delete:", width="stretch"):
            st.session_state['llm_config'] = LLMConfig()
            st.session_state['llm_models'] = []
            st.rerun()

    with right:
        ui.section("Supported providers")
        ui.card("".join(
            f'<div class="dia-feature"><span>{escape(name)}</span><b>{escape(p["env_var"])}</b></div>'
            for name, p in PROVIDERS.items()) +
            "<p style='margin-top:0.6rem'>Paste a key here for this session only, or put it in <code>.env</code>. "
            "OpenAI API access needs API billing; a ChatGPT subscription does not include it.</p>",
            title="Bring your own key")
        st.markdown("")
        ui.card("<p>• Keys are never written to disk, logged, or shown back.</p>"
                "<p>• Errors are scrubbed so a key can never appear in a message.</p>"
                "<p>• Only summary statistics are sent to the provider, never raw rows.</p>",
                title="Privacy & security")

    if st.session_state['llm_usage']:
        ui.section("Usage this session", "Token counts as reported by the provider; blank means not reported.")
        st.dataframe(pd.DataFrame(st.session_state['llm_usage']), width="stretch", hide_index=True)
