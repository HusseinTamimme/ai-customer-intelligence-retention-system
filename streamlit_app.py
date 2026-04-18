from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence & Retention System",
    page_icon="📊",
    layout="wide",
)

HIGH_RISK = 0.60
HIGH_VALUE = 0.70

DEFAULT_DATA_FILES = [
    "business_ready_data.csv",
    "final_scored_data.csv",
    "modeled_churn_data.csv",
    "segmented_data.csv",
]


@st.cache_data
def load_project_data(uploaded_file=None):
    """
    Priority:
    1) Uploaded CSV/XLSX from the sidebar
    2) Project files committed in the repo
    """
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file), f"Uploaded file: {uploaded_file.name}"
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file), f"Uploaded file: {uploaded_file.name}"
        raise ValueError("Please upload a CSV or Excel file.")

    for file_name in DEFAULT_DATA_FILES:
        path = Path(file_name)
        if path.exists():
            return pd.read_csv(path), f"Loaded from repo: {file_name}"

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
            df["customer_id_display"] = df.index.astype(str)
    return df


def metric_value(df, column, default=0):
    if column not in df.columns:
        return default
    series = pd.to_numeric(df[column], errors="coerce")
    return float(series.mean()) if series.notna().any() else default


def metric_sum(df, column, mask=None, default=0):
    if column not in df.columns:
        return default
    series = pd.to_numeric(df[column], errors="coerce")
    if mask is None:
        return float(series.sum()) if series.notna().any() else default
    return float(series.loc[mask].sum()) if series.loc[mask].notna().any() else default


@st.cache_data
def compute_histogram_counts(series: pd.Series, bins: int = 25) -> pd.DataFrame:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.DataFrame(columns=["Bin", "Count"])

    binned = pd.cut(clean, bins=bins, include_lowest=True)
    counts = binned.value_counts().sort_index()
    hist_df = counts.reset_index()
    hist_df.columns = ["Bin", "Count"]
    hist_df["Bin"] = hist_df["Bin"].astype(str)
    return hist_df


@st.cache_data
def compute_churn_distribution_by_actual(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    required = {"Churn", "churn_probability"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["Probability Bin", "Stayed", "Churned"])

    temp = df[["Churn", "churn_probability"]].copy()
    temp["churn_probability"] = pd.to_numeric(temp["churn_probability"], errors="coerce")
    temp["Churn"] = pd.to_numeric(temp["Churn"], errors="coerce")
    temp = temp.dropna(subset=["churn_probability", "Churn"])
    if temp.empty:
        return pd.DataFrame(columns=["Probability Bin", "Stayed", "Churned"])

    temp["Actual Status"] = temp["Churn"].map({0: "Stayed", 1: "Churned"}).fillna("Unknown")
    temp["Probability Bin"] = pd.cut(temp["churn_probability"], bins=bins, include_lowest=True)

    grouped = (
        temp.groupby(["Probability Bin", "Actual Status"], observed=False)
        .size()
        .reset_index(name="Count")
    )
    if grouped.empty:
        return pd.DataFrame(columns=["Probability Bin", "Stayed", "Churned"])

    grouped["Probability Bin"] = grouped["Probability Bin"].astype(str)
    pivot_df = grouped.pivot(index="Probability Bin", columns="Actual Status", values="Count").fillna(0)
    for col in ["Stayed", "Churned"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    pivot_df = pivot_df[["Stayed", "Churned"]].reset_index()
    return pivot_df


@st.cache_data
def get_category_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_category" not in df.columns:
        return pd.DataFrame(columns=["Customer Category", "Count"])

    category_df = (
        df["customer_category"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    category_df.columns = ["Customer Category", "Count"]
    return category_df[category_df["Count"] > 0]


@st.cache_data
def get_actions_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    if "recommended_action" not in df.columns:
        return pd.DataFrame(columns=["recommended_action", "Customers"])

    actions_df = (
        df["recommended_action"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    actions_df.columns = ["recommended_action", "Customers"]
    return actions_df


def render_scatter_chart(df: pd.DataFrame):
    if not {"churn_probability", "value_score"}.issubset(df.columns):
        st.warning("The required columns for the scatter chart were not found.")
        return

    scatter_df = df[["churn_probability", "value_score"]].copy()
    scatter_df["churn_probability"] = pd.to_numeric(scatter_df["churn_probability"], errors="coerce")
    scatter_df["value_score"] = pd.to_numeric(scatter_df["value_score"], errors="coerce")
    scatter_df = scatter_df.dropna()

    if scatter_df.empty:
        st.warning("No valid data available for the scatter chart.")
        return

    st.vega_lite_chart(
        scatter_df,
        {
            "layer": [
                {
                    "mark": {"type": "point", "filled": True, "opacity": 0.35, "size": 45},
                    "encoding": {
                        "x": {
                            "field": "churn_probability",
                            "type": "quantitative",
                            "title": "Churn Probability",
                            "scale": {"domain": [0, 1]},
                        },
                        "y": {
                            "field": "value_score",
                            "type": "quantitative",
                            "title": "Value Score",
                            "scale": {"domain": [0, 1.05]},
                        },
                    },
                },
                {
                    "mark": {"type": "rule", "strokeDash": [6, 4], "strokeWidth": 2},
                    "encoding": {"x": {"datum": HIGH_RISK}},
                },
                {
                    "mark": {"type": "rule", "strokeDash": [6, 4], "strokeWidth": 2},
                    "encoding": {"y": {"datum": HIGH_VALUE}},
                },
            ],
            "height": 380,
            "title": "Customer Risk vs Value",
        },
        use_container_width=True,
    )


def render_category_pie_chart(category_df: pd.DataFrame):
    if category_df.empty or category_df["Count"].sum() == 0:
        st.warning("No data available for Customer Category Distribution.")
        return

    color_domain = [
        "Loyal High Value",
        "At-Risk High Value",
        "Stable Customer",
        "At-Risk Low Value",
        "High Value - Low Risk",
        "High Value - High Risk",
        "Low Value - Low Risk",
        "Low Value - High Risk",
        "Unknown",
    ]
    color_range = [
        "#2ECC71",
        "#E74C3C",
        "#3498DB",
        "#F39C12",
        "#2ECC71",
        "#E74C3C",
        "#3498DB",
        "#F39C12",
        "#9AA0A6",
    ]

    st.vega_lite_chart(
        category_df,
        {
            "layer": [
                {
                    "mark": {"type": "arc", "innerRadius": 65, "outerRadius": 135},
                    "encoding": {
                        "theta": {"field": "Count", "type": "quantitative"},
                        "color": {
                            "field": "Customer Category",
                            "type": "nominal",
                            "legend": {"title": "Customer Category"},
                            "scale": {"domain": color_domain, "range": color_range},
                        },
                        "tooltip": [
                            {"field": "Customer Category", "type": "nominal"},
                            {"field": "Count", "type": "quantitative"},
                        ],
                    },
                },
                {
                    "transform": [
                        {
                            "joinaggregate": [{"op": "sum", "field": "Count", "as": "Total"}],
                        },
                        {
                            "calculate": "datum.Count / datum.Total",
                            "as": "Share",
                        },
                        {
                            "filter": "datum.Share >= 0.05",
                        },
                        {
                            "calculate": "format(datum.Share, '.0%')",
                            "as": "PercentLabel",
                        },
                    ],
                    "mark": {"type": "text", "radius": 165, "fontSize": 12},
                    "encoding": {
                        "theta": {"field": "Count", "type": "quantitative"},
                        "text": {"field": "PercentLabel"},
                    },
                },
            ],
            "height": 380,
            "title": "Customer Category Distribution",
            "view": {"stroke": None},
        },
        use_container_width=True,
    )


def get_customer_view(df: pd.DataFrame, customer_id: str):
    row = df.loc[df["customer_id_display"] == customer_id]
    if row.empty:
        return pd.DataFrame(), pd.DataFrame()

    row = row.iloc[0]

    info = pd.DataFrame(
        {
            "Field": [
                "Customer ID",
                "Segment",
                "Customer Category",
                "Churn Probability",
                "Value Score",
                "Recommended Action",
            ],
            "Value": [
                row.get("customer_id_display", "N/A"),
                row.get("segment", "N/A"),
                row.get("customer_category", "N/A"),
                f"{row.get('churn_probability', 0):.1%}" if pd.notna(row.get("churn_probability", None)) else "N/A",
                f"{row.get('value_score', 0):.2f}" if pd.notna(row.get("value_score", None)) else "N/A",
                row.get("recommended_action", "N/A"),
            ],
        }
    )

    snapshot_cols = [
        c
        for c in [
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
            "Tenure Months",
        ]
        if c in df.columns
    ]

    snapshot = row[snapshot_cols].to_frame("Value").reset_index() if snapshot_cols else pd.DataFrame()
    if not snapshot.empty:
        snapshot.columns = ["Feature", "Value"]

    return info, snapshot


def get_business_view(df: pd.DataFrame):
    category_df = get_category_chart_df(df)
    actions_df = get_actions_chart_df(df)
    top_risk = pd.DataFrame()

    sort_cols = [c for c in ["churn_probability", "value_score"] if c in df.columns]
    top_cols = [
        c
        for c in [
            "customer_id_display",
            "segment",
            "churn_probability",
            "value_score",
            "recommended_action",
            "Total Charges",
        ]
        if c in df.columns
    ]

    if sort_cols and top_cols:
        ascending = [False] * len(sort_cols)
        top_risk = df.sort_values(sort_cols, ascending=ascending).head(10)[top_cols]

    return category_df, actions_df, top_risk


def get_model_insights(df: pd.DataFrame):
    cols = [
        c
        for c in [
            "Contract",
            "Monthly Charges",
            "Total Charges",
            "Tenure Months",
            "service_count",
            "engagement_score",
        ]
        if c in df.columns
    ]
    stats = df[cols].describe(include="all").transpose() if cols else pd.DataFrame()
    prob_dist_df = compute_churn_distribution_by_actual(df)
    return stats, prob_dist_df


st.title("AI Customer Intelligence & Retention System")
st.caption("")

with st.sidebar:
    st.markdown("**Expected repo files:**")
    st.code("\n".join(DEFAULT_DATA_FILES))
    st.markdown("**Thresholds**")
    st.write(f"High-risk threshold: `{HIGH_RISK}`")
    st.write(f"High-value threshold: `{HIGH_VALUE}`")

try:
    df, source_note = load_project_data(uploaded_file)
    df = ensure_customer_id(df)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(source_note)

total_customers = len(df)
churn_rate = metric_value(df, "Churn", 0.0)
high_risk = int((pd.to_numeric(df["churn_probability"], errors="coerce") >= HIGH_RISK).sum()) if "churn_probability" in df.columns else 0
avg_value = metric_value(df, "value_score", 0.0)
revenue_at_risk = (
    metric_sum(df, "Total Charges", pd.to_numeric(df["churn_probability"], errors="coerce") >= HIGH_RISK, 0.0)
    if {"churn_probability", "Total Charges"}.issubset(df.columns)
    else 0.0
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churn Rate", f"{churn_rate:.1%}")
col3.metric("High-Risk Customers", f"{high_risk:,}")
col4.metric("Average Value Score", f"{avg_value:.2f}")
col5.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")

tabs = st.tabs(["Overview", "Customer Lookup", "Business View", "Model Insights", "Export"])

with tabs[0]:
    st.subheader("Overview")
    st.info("This app is locked to the Telco project outputs.")

    overview_left, overview_right = st.columns(2)

    with overview_left:
        render_scatter_chart(df)

    with overview_right:
        category_df = get_category_chart_df(df)
        render_category_pie_chart(category_df)

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.markdown("**Recommended Actions**")
        actions_df = get_actions_chart_df(df)
        if not actions_df.empty:
            st.bar_chart(actions_df.set_index("recommended_action"), horizontal=True, use_container_width=True)
        else:
            st.warning("The required recommended action column was not found in the dataset.")

    with bottom_right:
        st.markdown("**Churn Probability Distribution**")
        if "churn_probability" in df.columns:
            hist_df = compute_histogram_counts(df["churn_probability"])
            if not hist_df.empty:
                st.bar_chart(hist_df.set_index("Bin"), use_container_width=True)
                st.caption(f"Reference thresholds: high risk = {HIGH_RISK:.2f}, high value = {HIGH_VALUE:.2f}")
            else:
                st.warning("No valid churn probability data available.")
        else:
            st.warning("The required churn probability column was not found in the dataset.")

with tabs[1]:
    st.subheader("Customer Lookup")
    customer_choices = df["customer_id_display"].astype(str).tolist()
    selected_customer = st.selectbox("Select customer", customer_choices)

    info_df, snapshot_df = get_customer_view(df, selected_customer)
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("**Customer Summary**")
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("**Customer Snapshot**")
        st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Business View")
    category_df, actions_df, top_risk_df = get_business_view(df)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Customer Category Distribution**")
        if not category_df.empty:
            st.bar_chart(category_df.set_index("Customer Category"), horizontal=True, use_container_width=True)
        else:
            st.dataframe(category_df, use_container_width=True, hide_index=True)
    with b2:
        st.markdown("**Recommended Actions Distribution**")
        if not actions_df.empty:
            st.bar_chart(actions_df.set_index("recommended_action"), horizontal=True, use_container_width=True)
        else:
            st.dataframe(actions_df, use_container_width=True, hide_index=True)

    st.markdown("**Top Customers to Save**")
    st.dataframe(top_risk_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Model Insights")
    stats_df, prob_dist_df = get_model_insights(df)

    st.markdown("**Feature Snapshot**")
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("**Probability Distribution by Actual Churn**")
    if not prob_dist_df.empty:
        st.bar_chart(prob_dist_df.set_index("Probability Bin"), use_container_width=True)
    else:
        st.warning("Model insight chart needs both 'Churn' and 'churn_probability' columns.")

with tabs[4]:
    st.subheader("Export")
    export_df = df.copy()
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name="customer_intelligence_results.csv",
        mime="text/csv",
    )
