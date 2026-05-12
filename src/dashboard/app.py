"""
Streamlit dashboard for the BDT Financial Distress Early Warning System.

Reads gold Parquet files via DuckDB and visualises:
  - Pipeline overview (row counts, zone distribution)
  - Altman Z-Score distribution
  - Per-company Z-Score timeline
  - QoQ trend features
  - Distressed vs healthy company comparison
"""

import os
import glob
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
GOLD_LOCAL    = str(PROJECT_ROOT / "data" / "cache" / "gold_distress")
SILVER_LOCAL  = str(PROJECT_ROOT / "data" / "cache" / "silver_financials")
TEXT_LOCAL    = str(PROJECT_ROOT / "data" / "cache" / "silver_text")
UNIVERSE_PATH = PROJECT_ROOT / "config" / "company_universe.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BDT Distress Early Warning",
    page_icon="📉",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_gold() -> pd.DataFrame:
    files = glob.glob(f"{GOLD_LOCAL}/**/*.parquet", recursive=True)
    if not files:
        return pd.DataFrame()
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{GOLD_LOCAL}/**/*.parquet', hive_partitioning=true, union_by_name=true)"
    ).df()
    con.close()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    return df

@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_PATH, dtype={"cik": str})


@st.cache_data
def load_text_features() -> pd.DataFrame:
    """
    Load LLM text features from local cache (data/cache/silver_text).
    Falls back to downloading from MinIO silver bucket if cache is missing.
    Returns an empty DataFrame if neither source is available.
    """
    local_files = glob.glob(f"{TEXT_LOCAL}/**/*.parquet", recursive=True)
    if local_files:
        con = duckdb.connect()
        df = con.execute(
            f"SELECT * FROM read_parquet('{TEXT_LOCAL}/**/*.parquet', union_by_name=true)"
        ).df()
        con.close()
        return df

    # Try MinIO fallback
    try:
        from minio import Minio
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false") == "true",
        )
        silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
        objects = list(client.list_objects(
            silver_bucket, prefix="edgar_text_features/", recursive=True
        ))
        if not objects:
            return pd.DataFrame()

        import io, pyarrow.parquet as pq
        Path(TEXT_LOCAL).mkdir(parents=True, exist_ok=True)
        frames = []
        for obj in objects:
            if not obj.object_name.endswith(".parquet"):
                continue
            resp = client.get_object(silver_bucket, obj.object_name)
            try:
                buf = io.BytesIO(resp.read())
            finally:
                resp.close(); resp.release_conn()
            frames.append(pq.read_table(buf).to_pandas())
            # Cache locally for next time
            local_path = Path(TEXT_LOCAL) / Path(obj.object_name).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                buf.seek(0); f.write(buf.read())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


df = load_gold()
universe = load_universe()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📉 BDT Distress Warning")
st.sidebar.markdown("**Medallion pipeline:** Bronze → Silver → Gold")
st.sidebar.divider()

if df.empty:
    st.error("No gold data found. Run `python run_pipeline.py` first.")
    st.stop()

# Filters
form_options = sorted(df["form_type"].dropna().unique())
selected_forms = st.sidebar.multiselect("Filing type", form_options, default=["10-K"])

year_min = int(df["period_end"].dt.year.min())
year_max = int(df["period_end"].dt.year.max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (2010, year_max))

show_label = st.sidebar.radio(
    "Company type",
    ["All", "Healthy (S&P 500)", "Distressed (LoPucki)"],
)

st.sidebar.divider()
st.sidebar.caption(f"Gold rows loaded: {len(df):,}")

# LLM data availability indicator
text_df = load_text_features()
if text_df.empty:
    st.sidebar.info(
        "🤖 **LLM text analysis** not yet available.\n\n"
        "Run the text pipeline to unlock:\n"
        "```\npython run_pipeline.py --with-text\n```"
    )
else:
    st.sidebar.success(f"🤖 LLM features: {len(text_df):,} docs")

# Apply filters
filtered = df[df["form_type"].isin(selected_forms)]
filtered = filtered[
    (filtered["period_end"].dt.year >= year_range[0]) &
    (filtered["period_end"].dt.year <= year_range[1])
]
if show_label == "Healthy (S&P 500)":
    filtered = filtered[filtered["distress_label"] == 0]
elif show_label == "Distressed (LoPucki)":
    filtered = filtered[filtered["distress_label"] == 1]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📉 Corporate Financial Distress Early Warning")
st.caption("Big Data Technologies — University of Trento | Bronze → Silver → Gold Medallion Architecture")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

zone_counts = filtered["distress_zone"].value_counts()
n_companies = filtered["cik"].nunique()
n_rows = len(filtered)
n_with_z = filtered["altman_z_score"].notna().sum()
mean_z = filtered["altman_z_score"].mean()

col1.metric("Companies", f"{n_companies:,}")
col2.metric("Periods", f"{n_rows:,}")
col3.metric("Z-Scores computed", f"{n_with_z:,}")
col4.metric("Mean Z-Score", f"{mean_z:.2f}" if pd.notna(mean_z) else "—")
col5.metric("Distress zone %",
            f"{zone_counts.get('distress', 0) / max(zone_counts.sum(), 1) * 100:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Row 1: Zone distribution + Z-Score histogram
# ---------------------------------------------------------------------------
row1_l, row1_r = st.columns(2)

with row1_l:
    st.subheader("Distress Zone Distribution")
    zone_df = filtered["distress_zone"].value_counts().reset_index()
    zone_df.columns = ["zone", "count"]
    zone_color = {"distress": "#e74c3c", "grey": "#f39c12", "safe": "#2ecc71"}
    fig_zone = px.pie(
        zone_df, names="zone", values="count",
        color="zone", color_discrete_map=zone_color,
        hole=0.4,
    )
    fig_zone.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_zone, use_container_width=True)

with row1_r:
    st.subheader("Altman Z-Score Distribution")
    z_data = filtered["altman_z_score"].dropna()
    fig_hist = px.histogram(
        z_data, nbins=80, color_discrete_sequence=["#3498db"],
        labels={"value": "Z-Score", "count": "Periods"},
    )
    fig_hist.add_vline(x=1.81, line_dash="dash", line_color="#e74c3c",
                       annotation_text="Distress (1.81)",
                       annotation_position="top left",
                       annotation_font_color="#e74c3c")
    fig_hist.add_vline(x=2.99, line_dash="dash", line_color="#2ecc71",
                       annotation_text="Safe (2.99)",
                       annotation_position="top right",
                       annotation_font_color="#2ecc71")
    fig_hist.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Z-Score over time (healthy vs distressed)
# ---------------------------------------------------------------------------
st.subheader("Median Altman Z-Score Over Time — Healthy vs Distressed")

time_df = (
    filtered[filtered["altman_z_score"].notna()]
    .groupby(["period_end", "distress_label"])["altman_z_score"]
    .median()
    .reset_index()
)
time_df["Company type"] = time_df["distress_label"].map(
    {0: "Healthy (S&P 500)", 1: "Distressed (LoPucki)"}
)
fig_time = px.line(
    time_df, x="period_end", y="altman_z_score", color="Company type",
    color_discrete_map={
        "Healthy (S&P 500)": "#2ecc71",
        "Distressed (LoPucki)": "#e74c3c",
    },
    labels={"period_end": "Period", "altman_z_score": "Median Z-Score"},
)
fig_time.add_hline(y=1.81, line_dash="dot", line_color="#e74c3c", opacity=0.5)
fig_time.add_hline(y=2.99, line_dash="dot", line_color="#2ecc71", opacity=0.5)
fig_time.update_layout(margin=dict(t=20, b=20))
st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Per-company Z-Score timeline
# ---------------------------------------------------------------------------
st.subheader("Per-Company Z-Score Timeline")

company_options = (
    filtered[filtered["altman_z_score"].notna()]
    .groupby("company_name")["altman_z_score"].count()
    .sort_values(ascending=False)
    .head(100)
    .index.tolist()
)

selected_companies = st.multiselect(
    "Select companies to compare",
    company_options,
    default=company_options[:3] if len(company_options) >= 3 else company_options,
)

if selected_companies:
    company_df = filtered[
        filtered["company_name"].isin(selected_companies) &
        filtered["altman_z_score"].notna()
    ]
    fig_co = px.line(
        company_df, x="period_end", y="altman_z_score",
        color="company_name",
        labels={"period_end": "Period", "altman_z_score": "Z-Score", "company_name": "Company"},
        markers=True,
    )
    fig_co.add_hline(y=1.81, line_dash="dot", line_color="#e74c3c", opacity=0.5,
                     annotation_text="Distress threshold")
    fig_co.add_hline(y=2.99, line_dash="dot", line_color="#2ecc71", opacity=0.5,
                     annotation_text="Safe threshold")
    fig_co.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_co, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 4: Trend-break flags + QoQ growth
# ---------------------------------------------------------------------------
row4_l, row4_r = st.columns(2)

with row4_l:
    st.subheader("Trend-Break Flags")
    rev_decline = filtered["revenue_declining_3q"].sum()
    ni_decline = filtered["net_income_declining_3q"].sum()
    flag_df = pd.DataFrame({
        "Flag": ["Revenue declining 3Q+", "Net income declining 3Q+"],
        "Count": [int(rev_decline), int(ni_decline)],
    })
    fig_flags = px.bar(
        flag_df, x="Flag", y="Count",
        color="Flag",
        color_discrete_sequence=["#e67e22", "#e74c3c"],
    )
    fig_flags.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_flags, use_container_width=True)

with row4_r:
    st.subheader("Revenue Growth QoQ — Healthy vs Distressed")
    growth_df = filtered[filtered["revenue_growth_qoq"].notna()].copy()
    growth_df["Company type"] = growth_df["distress_label"].map(
        {0: "Healthy", 1: "Distressed"}
    )
    fig_growth = px.box(
        growth_df, x="Company type", y="revenue_growth_qoq",
        color="Company type",
        color_discrete_map={"Healthy": "#2ecc71", "Distressed": "#e74c3c"},
        labels={"revenue_growth_qoq": "Revenue Growth QoQ"},
        points=False,
    )
    fig_growth.update_layout(
        showlegend=False,
        yaxis_range=[-1, 1],
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_growth, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 5: LLM Text Analysis (optional — shown only when text pipeline has run)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🤖 LLM Text Analysis")

if text_df.empty:
    st.info(
        "LLM text features are not yet available. "
        "Run `python run_pipeline.py --with-text` after setting up Ollama "
        "(`ollama pull llama3.2:3b`) to unlock this section.",
        icon="💡",
    )
else:
    # Build a CIK → company_name lookup from the gold data
    cik_to_name = (
        df[["cik", "company_name"]].drop_duplicates()
        .assign(cik=lambda x: x["cik"].astype(str).str.lstrip("0"))
        .set_index("cik")["company_name"]
        .to_dict()
    )
    text_df["company_name"] = text_df["cik"].astype(str).str.lstrip("0").map(cik_to_name).fillna("Unknown")

    # ---- KPI strip ----
    tk1, tk2, tk3, tk4 = st.columns(4)
    n_docs       = len(text_df)
    n_cos_text   = text_df["cik"].nunique()
    pct_gc       = text_df["going_concern"].mean() * 100 if "going_concern" in text_df.columns else 0
    pct_neg      = (text_df["sentiment_score"] < -0.2).mean() * 100 if "sentiment_score" in text_df.columns else 0
    tk1.metric("Documents analysed", f"{n_docs:,}")
    tk2.metric("Companies covered",  f"{n_cos_text:,}")
    tk3.metric("Going-concern flags", f"{pct_gc:.1f}%")
    tk4.metric("Negative sentiment",  f"{pct_neg:.1f}%")

    txt_l, txt_r = st.columns(2)

    # ---- Sentiment score by section (box plot — shows real distribution) ----
    with txt_l:
        st.markdown("**Sentiment Score by Filing Section**")
        if "sentiment_score" in text_df.columns and "section" in text_df.columns:
            fig_sent = px.box(
                text_df, x="section", y="sentiment_score",
                color="section",
                color_discrete_map={"mda": "#3498db", "risk": "#e74c3c", "business": "#2ecc71"},
                points="all",
                hover_data=["company_name"] if "company_name" in text_df.columns else None,
                labels={"section": "Section", "sentiment_score": "Sentiment Score (−1 negative → +1 positive)"},
                range_y=[-1.1, 1.1],
            )
            fig_sent.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig_sent.update_layout(margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig_sent, use_container_width=True)

    # ---- Risk level distribution ----
    with txt_r:
        st.markdown("**Risk Level Distribution**")
        if "risk_level" in text_df.columns:
            risk_order  = ["low", "medium", "high", "critical"]
            risk_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e67e22", "critical": "#e74c3c"}
            risk_counts = text_df["risk_level"].value_counts().reindex(risk_order).dropna().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            fig_risk = px.bar(
                risk_counts, x="risk_level", y="count",
                color="risk_level", color_discrete_map=risk_colors,
                labels={"risk_level": "Risk Level", "count": "Documents"},
            )
            fig_risk.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_risk, use_container_width=True)

    # ---- Sentiment score vs Z-Score scatter (MD&A only) ----
    st.markdown("**Sentiment Score vs Altman Z-Score (MD&A section)**")
    mda_df = text_df[text_df["section"] == "mda"][["cik", "sentiment_score", "risk_level", "going_concern", "company_name"]].copy()
    mda_df["cik_str"] = mda_df["cik"].astype(str).str.lstrip("0")

    # Get latest Z-score per company from gold
    latest_z = (
        df[df["altman_z_score"].notna()]
        .sort_values("period_end", ascending=False)
        .drop_duplicates(subset="cik")
        [["cik", "altman_z_score", "distress_zone"]]
        .assign(cik_str=lambda x: x["cik"].astype(str).str.lstrip("0"))
    )
    scatter_df = mda_df.merge(latest_z, on="cik_str", how="inner")

    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x="altman_z_score", y="sentiment_score",
            color="distress_zone",
            color_discrete_map={"distress": "#e74c3c", "grey": "#f39c12", "safe": "#2ecc71"},
            hover_data={"company_name": True, "risk_level": True, "going_concern": True},
            labels={
                "altman_z_score":  "Altman Z-Score (latest)",
                "sentiment_score": "MD&A Sentiment Score",
                "distress_zone":   "Zone",
            },
            symbol="going_concern",
            symbol_map={0: "circle", 1: "x"},
        )
        fig_scatter.add_vline(x=1.81, line_dash="dot", line_color="#e74c3c", opacity=0.4)
        fig_scatter.add_vline(x=2.99, line_dash="dot", line_color="#2ecc71", opacity=0.4)
        fig_scatter.add_hline(y=0,    line_dash="dot", line_color="#94a3b8",  opacity=0.4)
        fig_scatter.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("✕ markers = going-concern language detected in the MD&A")
    else:
        st.info("Not enough overlap between text features and Z-score data to plot.")

    # ---- LLM summaries table ----
    with st.expander("📝 LLM Summaries — MD&A"):
        if "llm_summary" in text_df.columns:
            summary_df = (
                text_df[text_df["section"] == "mda"]
                [["company_name", "sentiment_label", "risk_level", "going_concern",
                  "distress_keywords", "llm_summary"]]
                .sort_values(["risk_level", "going_concern"], ascending=[False, False])
                .rename(columns={
                    "company_name":      "Company",
                    "sentiment_label":   "Sentiment",
                    "risk_level":        "Risk",
                    "going_concern":     "Going Concern",
                    "distress_keywords": "Distress Keywords",
                    "llm_summary":       "LLM Summary",
                })
            )
            st.dataframe(summary_df, use_container_width=True, height=350)

# ---------------------------------------------------------------------------
# Row 6: Raw data explorer
# ---------------------------------------------------------------------------
with st.expander("🔍 Raw financial data explorer"):
    cols_to_show = [
        "cik", "company_name", "period_end", "form_type",
        "altman_z_score", "distress_zone",
        "x1_wc_to_assets", "x2_re_to_assets", "x3_ebit_to_assets",
        "x4_equity_to_liab", "x5_rev_to_assets",
        "revenue_growth_qoq", "net_income_growth_qoq",
        "revenue_declining_3q", "net_income_declining_3q",
        "distress_label",
    ]
    available = [c for c in cols_to_show if c in filtered.columns]
    st.dataframe(
        filtered[available].sort_values("period_end", ascending=False),
        use_container_width=True,
        height=400,
    )

