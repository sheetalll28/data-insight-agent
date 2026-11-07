"""Headless run of the Streamlit app (no API key, no network)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "src", "app.py")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "messy_data.csv")


@pytest.fixture
def app(monkeypatch):
    for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(v, "")  # override anything a local .env would load
    import pandas as pd
    from pipeline import run_cleaning_pipeline
    at = AppTest.from_file(APP, default_timeout=120)
    clean, rep = run_cleaning_pipeline(pd.read_csv(DATA), return_report=True)
    at.session_state["raw_df"] = pd.read_csv(DATA)
    at.session_state["clean_df"] = clean
    at.session_state["clean_report"] = rep
    return at


@pytest.mark.parametrize("page", [":material/upload_file: Upload & Clean", ":material/query_stats: Feature Discovery",
                                  ":material/hub: Cohort Intelligence", ":material/tune: AI Settings"])
def test_pages_render_without_errors_or_api_key(app, page):
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]


def test_clustering_runs_without_api_key(app):
    app.run()
    app.sidebar.radio[0].set_value(":material/hub: Cohort Intelligence").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("Cluster Profiles" in m.value for m in app.markdown)
    assert any("optional" in i.value for i in app.info)


def test_feature_page_reports_held_out_r2(app):
    app.run()
    app.sidebar.radio[0].set_value(":material/query_stats: Feature Discovery").run()
    labels = [m.label for m in app.metric]
    assert "Held-out R²" in labels and "Training R²" in labels


def _configure_fake_llm(app, monkeypatch, json_reply=None, text_reply="**Cohort report**"):
    import llm_client
    meta = {"provider": "Groq", "model": "fake-model", "seconds": 0.01, "input_tokens": 5, "output_tokens": 5}
    monkeypatch.setattr(llm_client, "chat_json", lambda cfg, s, u, **kw: (json_reply, meta))
    monkeypatch.setattr(llm_client, "chat", lambda cfg, s, u, **kw: dict(meta, text=text_reply))
    app.session_state["llm_config"] = llm_client.LLMConfig(provider="Groq", api_key="fake", model="fake-model")


def test_ai_feature_suggestions_are_measured_not_trusted(app, monkeypatch):
    reply = {"recommendations": [
        {"feature_name": "Wellness", "raw_features_combined": ["Mood Score (1-10)", "Sleep Quality (1-10)"],
         "reasoning": "r"},
        {"feature_name": "Ghost", "raw_features_combined": ["Mood Score (1-10)", "Made Up Column"],
         "reasoning": "r"},
    ]}
    _configure_fake_llm(app, monkeypatch, json_reply=reply)
    app.run()
    app.sidebar.radio[0].set_value(":material/query_stats: Feature Discovery").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    table = app.dataframe[-1].value
    assert list(table["Proposed feature"]) == ["Wellness", "Ghost"]
    assert table.loc[0, "Measured |r| of composite"] is not None
    assert "Made Up Column" in table.loc[1, "Note"]          # hallucinated column rejected
    assert "expected_correlation_improvement_pct" not in table.columns


def test_ai_cohort_report_uses_configured_provider(app, monkeypatch):
    _configure_fake_llm(app, monkeypatch)
    app.run()
    app.sidebar.radio[0].set_value(":material/hub: Cohort Intelligence").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("Cohort report" in m.value for m in app.markdown)
    assert app.session_state["llm_usage"][-1]["model"] == "fake-model"


def test_bi_studio_renders_offline(app):
    pytest.importorskip("pygwalker")
    app.run()
    app.sidebar.radio[0].set_value(":material/dashboard: BI Studio").run()
    assert not app.exception and not app.error



def test_low_cardinality_target_uses_box_plots(app):
    app.run()
    app.sidebar.radio[0].set_value(":material/query_stats: Feature Discovery").run()
    app.selectbox[0].set_value("Outcome").run()
    assert not app.exception and not app.error
