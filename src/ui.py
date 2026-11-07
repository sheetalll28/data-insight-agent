"""
Presentation layer: theme, chart styling and small HTML components.
Keeps app.py focused on the analysis flow. No analytics happen here.

Chart colours follow a validated palette (dark-mode steps, checked against
the slate surface for colour-blind separation and contrast). Categorical
colours are assigned in fixed order and never cycled; clusters also get a
distinct marker shape so identity never relies on colour alone.
"""
from html import escape

import plotly.io as pio
import streamlit as st

# --- Design tokens -----------------------------------------------------------
BG = "#0f172a"
SURFACE = "#1e293b"
SURFACE_2 = "#162236"
BORDER = "#334155"
TEXT = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_FAINT = "#64748b"
ACCENT = "#818cf8"
ACCENT_2 = "#38bdf8"

# Categorical slots (dark steps), fixed order.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SYMBOLS = ["circle", "square", "diamond", "cross", "x", "circle-open", "square-open", "diamond-open"]
MAX_SERIES = len(SERIES)

# Diverging (negative red <-> neutral slate <-> positive blue) for correlations.
DIVERGING = [[0.0, "#e66767"], [0.5, "#475569"], [1.0, "#3987e5"]]
PRIMARY_SERIES = SERIES[0]
POSITIVE, NEGATIVE = "#3987e5", "#e66767"

# Status colours are reserved for state and always paired with an icon + label.
STATUS = {"good": ("#0ca30c", "▲"), "warning": ("#fab219", "●"), "critical": ("#d03b3b", "▼"),
          "neutral": (TEXT_FAINT, "●")}


def register_plotly_theme():
    tpl = pio.templates["plotly_dark"].to_plotly_json()
    layout = tpl["layout"]
    grid = "rgba(148,163,184,0.12)"
    axis = dict(gridcolor=grid, zerolinecolor="rgba(148,163,184,0.35)", linecolor="rgba(148,163,184,0.25)",
                tickfont=dict(color=TEXT_MUTED, size=12), title=dict(font=dict(color=TEXT_MUTED, size=12)))
    layout.update(
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left", y=0.97),
        xaxis=axis, yaxis=axis,
        legend=dict(font=dict(color=TEXT_MUTED, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, family="Inter, sans-serif")),
        margin=dict(l=10, r=10, t=48, b=10),
        coloraxis=dict(colorbar=dict(tickfont=dict(color=TEXT_MUTED), outlinewidth=0, thickness=10)),
    )
    for ax in ("xaxis", "yaxis", "zaxis"):
        layout.setdefault("scene", {})[ax] = dict(
            backgroundcolor="rgba(0,0,0,0)", gridcolor=grid, zerolinecolor=grid,
            tickfont=dict(color=TEXT_MUTED, size=10), title=dict(font=dict(color=TEXT_MUTED, size=11)))
    pio.templates["insight"] = tpl
    pio.templates.default = "insight"


def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Enterprise Deep Slate Background */
    .stApp {{ background-color: {BG}; color: #f8fafc; font-family: 'Inter', sans-serif; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

    h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }}
    h2, h3 {{ color: {TEXT} !important; font-weight: 600 !important; }}

    /* Page header */
    .dia-hero {{ padding: 0.2rem 0 1.2rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 1.4rem; }}
    .dia-hero .eyebrow {{ color: {ACCENT}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
                          text-transform: uppercase; margin-bottom: 0.35rem; }}
    .dia-hero h1 {{ color: {TEXT} !important; font-weight: 700 !important; font-size: 2.1rem !important;
                    margin: 0 !important; padding: 0 !important; line-height: 1.2; }}
    .dia-hero p {{ color: {TEXT_MUTED}; font-size: 1rem; margin: 0.45rem 0 0 0; max-width: 820px; }}

    /* Section titles */
    .dia-section {{ margin: 1.6rem 0 0.7rem 0; }}
    .dia-section .t {{ color: {TEXT}; font-size: 1.15rem; font-weight: 600; }}
    .dia-section .s {{ color: {TEXT_MUTED}; font-size: 0.9rem; margin-top: 0.15rem; }}

    /* Cards */
    .dia-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.1rem 1.2rem;
                 height: 100%; }}
    .dia-card h4 {{ color: {TEXT}; font-size: 1rem; font-weight: 600; margin: 0 0 0.35rem 0; }}
    .dia-card p, .dia-card li {{ color: {TEXT_MUTED}; font-size: 0.9rem; margin: 0.15rem 0; }}
    .dia-card .num {{ color: {ACCENT}; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.06em; }}
    .dia-card .big {{ color: {TEXT}; font-size: 1.5rem; font-weight: 700; }}
    .dia-dot {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:8px; vertical-align: 1px; }}
    .dia-badge {{ display:inline-flex; align-items:center; gap:6px; padding: 2px 10px; border-radius: 999px;
                  font-size: 0.78rem; font-weight: 600; border: 1px solid {BORDER}; background: {SURFACE_2};
                  color: {TEXT}; }}
    .dia-feature {{ display:flex; justify-content:space-between; gap: 10px; padding: 3px 0; font-size: 0.88rem;
                    color: {TEXT_MUTED}; border-bottom: 1px dashed rgba(148,163,184,0.12); }}
    .dia-feature span {{ min-width: 0; }}
    .dia-feature b {{ color: {TEXT}; font-weight: 600; white-space: nowrap; }}
    .dia-feature small {{ color: {TEXT_FAINT}; font-size: 0.78rem; }}
    .dia-report {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT};
                   border-radius: 12px; padding: 1.1rem 1.4rem; }}

    /* Professional Sidebar */
    [data-testid="stSidebar"] {{ background-color: {SURFACE} !important; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] .dia-brand {{ display:flex; align-items:center; gap:10px; margin: 0.2rem 0 1.2rem 0; }}
    [data-testid="stSidebar"] .dia-brand .logo {{ line-height: 0; }}
    /* Hide Streamlit's developer chrome (Deploy button, toolbar) */
    [data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stAppDeployButton,
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stSidebar"] .dia-brand .name {{ font-size: 1.05rem; font-weight: 700; line-height: 1.15;
        color: {TEXT}; }}
    [data-testid="stSidebar"] .dia-brand .tag {{ color: {TEXT_FAINT}; font-size: 0.75rem; }}
    [data-testid="stSidebar"] .dia-label {{ color: {TEXT_FAINT}; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase; margin: 1rem 0 0.4rem 0.2rem; }}

    /* Sidebar navigation: radio styled as menu items */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
    [data-testid="stSidebar"] [data-testid="stRadio"], [data-testid="stSidebar"] [data-testid="stRadio"] > div,
    [data-testid="stSidebar"] div[role="radiogroup"] {{ width: 100%; gap: 2px; display: flex;
        flex-direction: column; align-items: stretch; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label {{
        padding: 0.55rem 0.75rem; border-radius: 10px; margin: 0; width: 100%; max-width: none;
        box-sizing: border-box;
        transition: background 0.15s ease; cursor: pointer; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display: none; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {{ color: {TEXT_MUTED}; font-size: 0.93rem; font-weight: 500; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background: rgba(129,140,248,0.08); }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
        background: rgba(129,140,248,0.16); box-shadow: inset 3px 0 0 {ACCENT}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{ color: {TEXT}; font-weight: 600; }}

    .dia-step {{ display:flex; align-items:center; gap:8px; font-size: 0.85rem; color: {TEXT_MUTED}; padding: 3px 0.2rem; }}
    .dia-step .ic {{ width: 18px; height: 18px; border-radius: 50%; display:inline-flex; align-items:center;
                     justify-content:center; font-size: 0.7rem; font-weight: 700; }}
    .dia-step.done .ic {{ background: rgba(12,163,12,0.18); color: #4ade80; }}
    .dia-step.todo .ic {{ background: {SURFACE_2}; color: {TEXT_FAINT}; border: 1px solid {BORDER}; }}
    .dia-step.done {{ color: {TEXT}; }}

    /* Enterprise Buttons: gradient for the page's primary action, outline for the rest */
    .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {{
        background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
        padding: 0.55rem 1.2rem; font-weight: 600; border-radius: 10px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); }}
    .stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {{
        border-color: {ACCENT}; color: {TEXT}; background: rgba(129,140,248,0.10); }}
    .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primaryFormSubmit"] {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; border: none;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.35); }}
    .stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primaryFormSubmit"]:hover {{
        transform: translateY(-1px); box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.45); color: white;
        border: none; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); }}

    /* Metrics as cards */
    [data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
                                padding: 0.9rem 1.1rem; }}
    [data-testid="stMetricLabel"] p {{ color: {TEXT_MUTED} !important; font-size: 0.82rem !important; font-weight: 500; }}
    [data-testid="stMetricValue"] {{ color: {TEXT}; font-size: 1.6rem; font-weight: 700; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.55rem 1rem; border-radius: 10px 10px 0 0; color: {TEXT_MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {TEXT} !important; background: rgba(129,140,248,0.10); }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; }}

    /* Inputs, uploader, tables, expanders */
    div[data-baseweb="select"] > div {{ background-color: {SURFACE}; border: 1px solid {BORDER};
                                        border-radius: 10px; color: #f8fafc; }}
    [data-testid="stFileUploaderDropzone"] {{ background: {SURFACE}; border: 1.5px dashed #475569;
                                              border-radius: 14px; padding: 1.4rem; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
    [data-testid="stExpander"] details {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
    [data-testid="stPlotlyChart"] {{ border: 1px solid {BORDER}; border-radius: 14px;
                                     padding: 0.4rem 0.6rem; overflow: hidden; }}
    .dia-status {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin-bottom: 1rem; }}
    .dia-status .dia-card .v {{ color: {TEXT}; font-size: 1.05rem; font-weight: 600; word-break: break-all; margin-top: 0.3rem; }}
    .dia-status .dia-card .l {{ color: {TEXT_MUTED}; font-size: 0.82rem; }}
    [data-testid="stAlert"] {{ border-radius: 12px; }}
    hr {{ border-color: {BORDER} !important; }}
    </style>
    """, unsafe_allow_html=True)


# --- Components --------------------------------------------------------------

def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="dia-hero"><div class="eyebrow">{escape(eyebrow)}</div>'
                f'<h1>{escape(title)}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title, subtitle=""):
    sub = f'<div class="s">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="dia-section"><div class="t">{escape(title)}</div>{sub}</div>',
                unsafe_allow_html=True)


def badge(label, kind="neutral"):
    color, icon = STATUS.get(kind, STATUS["neutral"])
    return (f'<span class="dia-badge"><span style="color:{color}">{icon}</span>'
            f'{escape(str(label))}</span>')


def card(body_html, title="", number=""):
    num = f'<div class="num">{escape(number)}</div>' if number else ""
    head = f"<h4>{escape(title)}</h4>" if title else ""
    st.markdown(f'<div class="dia-card">{num}{head}{body_html}</div>', unsafe_allow_html=True)


def sidebar_brand():
    st.markdown('<div class="dia-brand"><div class="logo"><svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">'
                '<rect width="32" height="32" rx="8" fill="#4f46e5"/>'
                '<rect x="8" y="17" width="4" height="7" rx="1.5" fill="#c7d2fe"/>'
                '<rect x="14" y="12" width="4" height="12" rx="1.5" fill="#e0e7ff"/>'
                '<rect x="20" y="8" width="4" height="16" rx="1.5" fill="#ffffff"/></svg></div><div><div class="name">Data Insight Agent</div>'
                '<div class="tag">From-scratch ML · Any LLM</div></div></div>', unsafe_allow_html=True)


def sidebar_label(text):
    st.markdown(f'<div class="dia-label">{escape(text)}</div>', unsafe_allow_html=True)


def step(label, done):
    cls, ic = ("done", "✓") if done else ("todo", "")
    st.markdown(f'<div class="dia-step {cls}"><span class="ic">{ic}</span>{escape(label)}</div>',
                unsafe_allow_html=True)


def series_dot(i):
    return f'<span class="dia-dot" style="background:{SERIES[i % MAX_SERIES]}"></span>'


def status_grid(items):
    """items: list of (label, value_html). Wraps long values instead of truncating."""
    cells = "".join(f'<div class="dia-card"><div class="l">{escape(l)}</div><div class="v">{v}</div></div>'
                    for l, v in items)
    st.markdown(f'<div class="dia-status">{cells}</div>', unsafe_allow_html=True)


def chart(fig, container=None):
    """Render a Plotly figure full-width inside a bordered chart card."""
    (container or st).plotly_chart(fig, width="stretch", config={"displayModeBar": False, "displaylogo": False})
