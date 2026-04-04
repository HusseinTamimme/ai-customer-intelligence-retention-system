from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Customer Intelligence & Retention System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CONFIG ----------
HIGH_RISK = 0.60
HIGH_VALUE = 0.70
TOP_N = 10

DEFAULT_DATA_FILES = [
    "business_ready_data.csv",
    "final_scored_data.csv",
    "modeled_churn_data.csv",
    "segmented_data.csv",
]

# ---------- STYLING ----------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #fff8f8 0%, #ffffff 100%);
    }
    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 45%, #ef4444 100%);
        color: white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.95;
    }
    .section-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.7rem;
        color: #7f1d1d;
    }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #f3d2d2;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- DATA ----------
@st.cache_data
def load_project_data():
    for file_name in DEFAULT_DATA_FILES:
        path = Path(file_name)
        if path.exists():
            return pd.read_csv(path), f"Connected to {file_name}"
    raise FileNotFoundError(
        "No project data file found. Add one of these files to your repo root: "
        + ", ".join(DEFAULT_DATA_FILES)
    )


def ensure_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customer_id_display" not in df.columns:
        if "customerID" in df.columns:
            df["customer_id_display"] = df["customerID"].astype(str)
        else:
            df["customer_id_display"] = (
                "CUST-" + (df.index + 1).astype(str).str.zfill(5)
            )
    return df


def clean_action_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    return float(df[col].mean()) if col in df.columns else default


def safe_sum(df: pd.DataFrame, col: str, mask=None, default: float = 0.0) -> float:
    if col not in df.columns:
        return default
    if mask is None:
        return float(df[col].sum())
    return float(df.loc[mask, col].sum())


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------- CONSISTENT CHART THEME ----------
COLORS = {
    "primary": "#E50914",       # main brand red
    "secondary": "#B20710",     # darker red
    "accent": "#FF6B6B",        # soft red
    "light": "#FFD6D6",         # very light red
    "text": "#F9FAFB",          # near white
    "muted": "#9CA3AF",         # gray
    "grid": "#374151",          # dark gray
    "bg": "#0B1220",            # chart background
    "card": "#111827",          # panel background
}

RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "custom_reds",
    [COLORS["light"], COLORS["accent"], COLORS["primary"], COLORS["secondary"]]
)


def apply_chart_theme(fig, ax):
    fig.patch.set_facecolor(COLORS["card"])
    ax.set_facecolor(COLORS["card"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])

    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.22, color=COLORS["grid"])


def chart_style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color=COLORS["text"])
    ax.set_xlabel(xlabel, fontsize=10, color=COLORS["text"])
    ax.set_ylabel(ylabel, fontsize=10, color=COLORS["text"])


def add_value_labels_barh(ax, values, pad_ratio=0.012, money=False, pct=False):
    if len(values) == 0:
        return
    max_val = max(values) if max(values) != 0 else 1
    for i, v in enumerate(values):
        if money:
            label = f"${v:,.0f}"
        elif pct:
            label = f"{v:.0%}"
        else:
            label = f"{int(v):,}"
        ax.text(
            v + max_val * pad_ratio,
            i,
            label,
            va="center",
            fontsize=9,
            color=COLORS["text"]
        )


def add_value_labels_bar(ax, values, pad_ratio=0.02, pct=False, money=False):
    if len(values) == 0:
        return
    max_val = max(values) if max(values) != 0 else 1
    for i, v in enumerate(values):
        if money:
            label = f"${v:,.0f}"
        elif pct:
            label = f"{v:.0%}"
        else:
            label = f"{int(v):,}"
        ax.text(
            i,
            v + max_val * pad_ratio,
            label,
            ha="center",
            fontsize=9,
            color=COLORS["text"]
        )


def build_overview_charts(df: pd.DataFrame):
    charts = []

    # 1) Risk vs Value Map
    if {"churn_probability", "value_score"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5.2))

        if "Total Charges" in df.columns:
            sizes = df["Total Charges"].fillna(df["Total Charges"].median()).clip(lower=50) / 18
            sizes = np.clip(sizes, 30, 260)
        else:
            sizes = 80

        scatter = ax.scatter(
            df["churn_probability"],
            df["value_score"],
            c=df["churn_probability"],
            s=sizes,
            cmap=RED_CMAP,
            alpha=0.72,
            edgecolors="none"
        )

        ax.axvline(HIGH_RISK, color=COLORS["accent"], linestyle="--", linewidth=2)
        ax.axhline(HIGH_VALUE, color=COLORS["accent"], linestyle="--", linewidth=2)

        ax.text(
            HIGH_RISK + 0.015,
            df["value_score"].max() * 0.94 if len(df) else 0.9,
            "High Risk Zone",
            color=COLORS["text"],
            fontsize=9
        )
        ax.text(
            0.03,
            HIGH_VALUE + 0.02,
            "High Value Zone",
            color=COLORS["text"],
            fontsize=9
        )

        chart_style(
            ax,
            "Customer Risk vs Value Map",
            "Churn Probability",
            "Value Score"
        )
        apply_chart_theme(fig, ax)

        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Churn Probability", color=COLORS["text"])
        cbar.ax.yaxis.set_tick_params(color=COLORS["text"])
        plt.setp(cbar.ax.get_yticklabels(), color=COLORS["text"])

        charts.append(fig)

    # 2) Segment Distribution
    if "segment" in df.columns:
        seg = df["segment"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.8))

        bar_colors = [RED_CMAP(x) for x in np.linspace(0.35, 0.95, len(seg))]
        ax.barh(seg.index, seg.values, color=bar_colors)

        chart_style(ax, "Customer Segment Distribution", "Customers", "")
        apply_chart_theme(fig, ax)
        add_value_labels_barh(ax, seg.values)

        charts.append(fig)

    # 3) Revenue at Risk by Segment
    if {"segment", "Total Charges", "churn_probability"}.issubset(df.columns):
        temp = df.copy()
        temp = temp[temp["churn_probability"] >= HIGH_RISK].copy()

        if not temp.empty:
            rev = temp.groupby("segment")["Total Charges"].sum().sort_values()
            fig, ax = plt.subplots(figsize=(8, 4.8))

            bar_colors = [RED_CMAP(x) for x in np.linspace(0.4, 1.0, len(rev))]
            ax.barh(rev.index, rev.values, color=bar_colors)

            chart_style(ax, "Revenue at Risk by Segment", "Revenue at Risk", "")
            apply_chart_theme(fig, ax)
            add_value_labels_barh(ax, rev.values, money=True)

            charts.append(fig)

    # 4) Average Churn Risk by Contract Type
    if {"Contract", "churn_probability"}.issubset(df.columns):
        contract_risk = df.groupby("Contract")["churn_probability"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.8))

        bar_colors = [RED_CMAP(x) for x in np.linspace(0.45, 0.95, len(contract_risk))]
        ax.bar(contract_risk.index, contract_risk.values, color=bar_colors)

        chart_style(ax, "Average Churn Risk by Contract Type", "", "Avg Churn Probability")
        apply_chart_theme(fig, ax)
        add_value_labels_bar(ax, contract_risk.values, pct=True)

        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", color=COLORS["text"])
        charts.append(fig)

    # 5) Recommended Action Mix
    if "recommended_action" in df.columns:
        action_counts = (
            df["recommended_action"]
            .astype(str)
            .str.strip()
            .value_counts()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(8, 4.8))
        bar_colors = [RED_CMAP(x) for x in np.linspace(0.35, 0.95, len(action_counts))]
        ax.barh(action_counts.index, action_counts.values, color=bar_colors)

        chart_style(ax, "Recommended Action Mix", "Customers", "")
        apply_chart_theme(fig, ax)
        add_value_labels_barh(ax, action_counts.values)

        charts.append(fig)

    # 6) Churn Probability Distribution
    if "churn_probability" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.8))

        n, bins, patches = ax.hist(
            df["churn_probability"],
            bins=30,
            edgecolor="none"
        )

        for patch, left_edge in zip(patches, bins[:-1]):
            normalized = min(max(left_edge, 0), 1)
            patch.set_facecolor(RED_CMAP(normalized))
            patch.set_alpha(0.95)

        ax.axvline(HIGH_RISK, color=COLORS["accent"], linestyle="--", linewidth=2)
        ax.text(
            HIGH_RISK + 0.01,
            max(n) * 0.92 if len(n) else 0,
            "High-Risk Threshold",
            color=COLORS["text"],
            fontsize=9
        )

        chart_style(ax, "Churn Risk Distribution", "Churn Probability", "Customer Count")
        apply_chart_theme(fig, ax)

        charts.append(fig)

    return charts


def build_top_save_table(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"customer_id_display", "segment", "churn_probability", "value_score", "recommended_action"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    cols = [c for c in [
        "customer_id_display", "segment", "customer_category", "churn_probability",
        "value_score", "recommended_action", "Total Charges"
    ] if c in df.columns]

    top_df = df.sort_values(
        ["churn_probability", "value_score"],
        ascending=[False, False]
    ).head(TOP_N)[cols].copy()

    if "churn_probability" in top_df.columns:
        top_df["churn_probability"] = top_df["churn_probability"].map(lambda x: f"{x:.1%}")
    if "value_score" in top_df.columns:
        top_df["value_score"] = top_df["value_score"].map(lambda x: f"{x:.2f}")
    if "Total Charges" in top_df.columns:
        top_df["Total Charges"] = top_df["Total Charges"].map(lambda x: f"${x:,.0f}")

    return top_df


def get_customer_view(df: pd.DataFrame, customer_id: str):
    row = df.loc[df["customer_id_display"] == customer_id]
    if row.empty:
        return pd.DataFrame(), pd.DataFrame()

    row = row.iloc[0]

    summary = pd.DataFrame(
        {
            "Field": [
                "Customer ID",
                "Segment",
                "Customer Category",
                "Value Segment",
                "Churn Probability",
                "Predicted Value",
                "Recommended Action",
            ],
            "Value": [
                row.get("customer_id_display", "N/A"),
                row.get("segment", "N/A"),
                row.get("customer_category", "N/A"),
                row.get("value_segment", "N/A"),
                f"{row.get('churn_probability', 0):.1%}" if pd.notna(row.get("churn_probability", None)) else "N/A",
                f"${row.get('predicted_value', 0):,.0f}" if pd.notna(row.get("predicted_value", None)) else "N/A",
                str(row.get("recommended_action", "N/A")).strip(),
            ],
        }
    )

    snapshot_cols = [
        c for c in [
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Contract",
            "Payment Method",
            "Monthly Charges",
            "Total Charges",
            "service_count",
            "engagement_score",
            "tenure_group",
        ] if c in df.columns
    ]

    snapshot = row[snapshot_cols].to_frame("Value").reset_index() if snapshot_cols else pd.DataFrame()
    if not snapshot.empty:
        snapshot.columns = ["Feature", "Value"]

    return summary, snapshot


def get_business_tables(df: pd.DataFrame):
    tables = {}

    if "customer_category" in df.columns:
        t = df["customer_category"].value_counts().reset_index()
        t.columns = ["Customer Category", "Count"]
        tables["categories"] = t

    if "recommended_action" in df.columns:
        t = clean_action_series(df["recommended_action"]).value_counts().reset_index()
        t.columns = ["Recommended Action", "Count"]
        tables["actions"] = t

    if {"segment", "churn_probability", "value_score", "Total Charges"}.issubset(df.columns):
        segment_perf = (
            df.groupby("segment")
            .agg(
                customers=("segment", "size"),
                avg_churn_probability=("churn_probability", "mean"),
                avg_value_score=("value_score", "mean"),
                total_revenue=("Total Charges", "sum"),
            )
            .reset_index()
        )
        segment_perf["avg_churn_probability"] = segment_perf["avg_churn_probability"].map(lambda x: f"{x:.1%}")
        segment_perf["avg_value_score"] = segment_perf["avg_value_score"].map(lambda x: f"{x:.2f}")
        segment_perf["total_revenue"] = segment_perf["total_revenue"].map(lambda x: f"${x:,.0f}")
        tables["segment_perf"] = segment_perf

    tables["top_save"] = build_top_save_table(df)
    return tables


def get_model_insights(df: pd.DataFrame):
    stats = pd.DataFrame()
    fig = None

    cols = [c for c in [
        "Monthly Charges", "Total Charges", "service_count",
        "engagement_score", "churn_probability", "value_score"
    ] if c in df.columns]

    if cols:
        stats = df[cols].describe().transpose().round(2)

    if {"Churn", "churn_probability"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(df.loc[df["Churn"] == 0, "churn_probability"], bins=25, alpha=0.65, label="Stayed")
        ax.hist(df.loc[df["Churn"] == 1, "churn_probability"], bins=25, alpha=0.65, label="Churned")
        ax.axvline(HIGH_RISK, linestyle="--", linewidth=1.5)
        chart_style(ax, "Predicted Risk vs Actual Churn", "Churn Probability", "Customer Count")
        ax.legend(frameon=False)
    return stats, fig


# ---------- LOAD ----------
try:
    df, source_note = load_project_data()
    df = ensure_customer_id(df)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------- HERO ----------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">AI Customer Intelligence & Retention System</div>
        <div class="hero-sub">
            A recruiter-ready analytics product that predicts churn, scores customer value,
            segments customers, and recommends retention actions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(source_note)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## Control Center")
    st.success("Project dataset connected")
    st.markdown("### Risk Thresholds")
    st.write(f"High-risk threshold: `{HIGH_RISK}`")
    st.write(f"High-value threshold: `{HIGH_VALUE}`")

    if "segment" in df.columns:
        segment_options = ["All"] + sorted(df["segment"].dropna().astype(str).unique().tolist())
    else:
        segment_options = ["All"]

    selected_segment = st.selectbox("Filter by segment", segment_options)

    if "Contract" in df.columns:
        contract_options = ["All"] + sorted(df["Contract"].dropna().astype(str).unique().tolist())
    else:
        contract_options = ["All"]

    selected_contract = st.selectbox("Filter by contract", contract_options)

# ---------- FILTER ----------
filtered_df = df.copy()

if selected_segment != "All" and "segment" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["segment"].astype(str) == selected_segment]

if selected_contract != "All" and "Contract" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Contract"].astype(str) == selected_contract]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ---------- KPIs ----------
total_customers = len(filtered_df)
churn_rate = float(filtered_df["Churn"].mean()) if "Churn" in filtered_df.columns else 0.0
high_risk_customers = int((filtered_df["churn_probability"] >= HIGH_RISK).sum()) if "churn_probability" in filtered_df.columns else 0
avg_value_score = safe_mean(filtered_df, "value_score", 0.0)
revenue_at_risk = (
    safe_sum(filtered_df, "Total Charges", filtered_df["churn_probability"] >= HIGH_RISK, 0.0)
    if {"churn_probability", "Total Charges"}.issubset(filtered_df.columns)
    else 0.0
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Churn Rate", fmt_pct(churn_rate))
k3.metric("High-Risk Customers", f"{high_risk_customers:,}")
k4.metric("Average Value Score", f"{avg_value_score:.2f}")
k5.metric("Revenue at Risk", fmt_money(revenue_at_risk))

tabs = st.tabs(["Overview", "Customer Lookup", "Business View", "Model Insights", "Export"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.markdown('<div class="section-label">Executive Overview</div>', unsafe_allow_html=True)

    charts = build_overview_charts(filtered_df)
    if charts:
        for i in range(0, len(charts), 2):
            c1, c2 = st.columns(2)
            with c1:
                if i < len(charts):
                    st.pyplot(charts[i], use_container_width=True)
                    plt.close(charts[i])
            with c2:
                if i + 1 < len(charts):
                    st.pyplot(charts[i + 1], use_container_width=True)
                    plt.close(charts[i + 1])

    if {"segment", "Total Charges", "churn_probability"}.issubset(filtered_df.columns):
        risky_rev = (filtered_df.loc[filtered_df["churn_probability"] >= HIGH_RISK, "Total Charges"].sum())
        top_segment = (
            filtered_df.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0]
        )
    else:
        risky_rev = 0
        top_segment = "N/A"

    high_value_high_risk = 0
    if {"churn_probability", "value_score"}.issubset(filtered_df.columns):
        high_value_high_risk = int(
            ((filtered_df["churn_probability"] >= HIGH_RISK) & (filtered_df["value_score"] >= HIGH_VALUE)).sum()
        )

    st.markdown("### Strategic Insights")
    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            f"""
            <div class="insight-box">
            <b>Revenue Exposure</b><br>
            Estimated revenue at risk is <b>{fmt_money(risky_rev)}</b>, highlighting where retention campaigns can protect business value.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i2:
        st.markdown(
            f"""
            <div class="insight-box">
            <b>Priority Segment</b><br>
            <b>{top_segment}</b> currently shows the highest average churn risk and should be the first business focus.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i3:
        st.markdown(
            f"""
            <div class="insight-box">
            <b>Critical Customers</b><br>
            <b>{high_value_high_risk:,}</b> customers sit in the high-value, high-risk zone — the most important group to save.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- CUSTOMER LOOKUP ----------
with tabs[1]:
    st.markdown('<div class="section-label">Customer Intelligence View</div>', unsafe_allow_html=True)

    customer_choices = filtered_df["customer_id_display"].astype(str).tolist()
    selected_customer = st.selectbox("Select customer", customer_choices)

    summary_df, snapshot_df = get_customer_view(filtered_df, selected_customer)

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("#### Customer Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Customer Snapshot")
        st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

# ---------- BUSINESS VIEW ----------
with tabs[2]:
    st.markdown('<div class="section-label">Business Decision View</div>', unsafe_allow_html=True)

    tables = get_business_tables(filtered_df)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### Customer Category Distribution")
        st.dataframe(tables.get("categories", pd.DataFrame()), use_container_width=True, hide_index=True)

    with b2:
        st.markdown("#### Recommended Action Distribution")
        st.dataframe(tables.get("actions", pd.DataFrame()), use_container_width=True, hide_index=True)

    st.markdown("#### Segment Performance")
    st.dataframe(tables.get("segment_perf", pd.DataFrame()), use_container_width=True, hide_index=True)

    st.markdown("#### Top Customers to Save")
    st.dataframe(tables.get("top_save", pd.DataFrame()), use_container_width=True, hide_index=True)

# ---------- EXPORT ----------
with tabs[4]:
    st.markdown('<div class="section-label">Export Results</div>', unsafe_allow_html=True)
    export_df = filtered_df.copy()
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Results CSV",
        data=csv_data,
        file_name="customer_intelligence_results.csv",
        mime="text/csv",
    )

    st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)
