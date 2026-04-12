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

GOLD_LOCAL  = "/tmp/gold_distress"
SILVER_LOCAL = "/tmp/silver_financials"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
        f"SELECT * FROM read_parquet('{GOLD_LOCAL}/**/*.parquet', hive_partitioning=true)"
    ).df()
    con.close()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    return df

@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_PATH, dtype={"cik": str})


df = load_gold()
universe = load_universe()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📉 BDT Distress Warning")
st.sidebar.markdown("**Medallion pipeline:** Bronze → Silver → Gold")
st.sidebar.divider()

if df.empty:
    st.error("No gold data found at /tmp/gold_distress. Run `python src/gold/build_gold.py` first.")
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
# Row 5: Raw data explorer
# ---------------------------------------------------------------------------
with st.expander("🔍 Raw data explorer"):
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
