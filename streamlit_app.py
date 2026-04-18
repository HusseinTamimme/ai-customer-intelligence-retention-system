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
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file), f"Uploaded file: {uploaded_file.name}"
        if name.endswith((".xlsx", ".xls")):
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


@st.cache_data
def compute_histogram_counts(series: pd.Series, bins: int = 25) -> pd.DataFrame:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.DataFrame(columns=["bin_label", "count"])

    binned = pd.cut(clean, bins=bins, include_lowest=True)
    categories = binned.cat.categories
    ordered_counts = binned.value_counts().reindex(categories, fill_value=0)

    hist_df = ordered_counts.reset_index()
    hist_df.columns = ["bin", "count"]
    hist_df["bin_label"] = hist_df["bin"].astype(str)
    return hist_df[["bin_label", "count"]]


@st.cache_data
def compute_churn_distribution_by_actual(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    required = {"Churn", "churn_probability"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    temp = df[["Churn", "churn_probability"]].copy()
    temp["churn_probability"] = pd.to_numeric(temp["churn_probability"], errors="coerce")
    temp["Churn"] = pd.to_numeric(temp["Churn"], errors="coerce")
    temp = temp.dropna(subset=["churn_probability", "Churn"])
    if temp.empty:
        return pd.DataFrame()

    temp["actual_status"] = temp["Churn"].map({0: "Stayed", 1: "Churned"}).fillna("Unknown")
    temp["probability_bin"] = pd.cut(temp["churn_probability"], bins=bins, include_lowest=True)

    chart_df = (
        temp.groupby(["probability_bin", "actual_status"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    chart_df["probability_bin"] = chart_df["probability_bin"].astype(str)
    return chart_df


@st.cache_data
def compute_scatter_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"churn_probability", "value_score"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    scatter_df = df[["churn_probability", "value_score"]].copy()
    scatter_df["churn_probability"] = pd.to_numeric(scatter_df["churn_probability"], errors="coerce")
    scatter_df["value_score"] = pd.to_numeric(scatter_df["value_score"], errors="coerce")
    return scatter_df.dropna().reset_index(drop=True)


@st.cache_data
def compute_category_counts(df: pd.DataFrame, column: str, index_name: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()

    count_df = df[column].fillna("Unknown").astype(str).value_counts().reset_index()
    count_df.columns = [index_name, "Count"]
    return count_df


def ensure_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customer_id_display" not in df.columns:
        if "customerID" in df.columns:
            df["customer_id_display"] = df["customerID"].astype(str)
        else:
            df["customer_id_display"] = df.index.astype(str)
    return df


def metric_value(df: pd.DataFrame, column: str, default=0.0) -> float:
    if column not in df.columns:
        return float(default)
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float(default)


def metric_sum(df: pd.DataFrame, column: str, mask=None, default=0.0) -> float:
    if column not in df.columns:
        return float(default)
    numeric_col = pd.to_numeric(df[column], errors="coerce")
    if mask is None:
        numeric_col = numeric_col.dropna()
        return float(numeric_col.sum()) if not numeric_col.empty else float(default)

    filtered = numeric_col.loc[mask].dropna()
    return float(filtered.sum()) if not filtered.empty else float(default)


def get_customer_view(df: pd.DataFrame, customer_id: str):
    row = df.loc[df["customer_id_display"] == customer_id]
    if row.empty:
        return pd.DataFrame(), pd.DataFrame()

    row = row.iloc[0]

    churn_prob = pd.to_numeric(pd.Series([row.get("churn_probability")]), errors="coerce").iloc[0]
    value_score = pd.to_numeric(pd.Series([row.get("value_score")]), errors="coerce").iloc[0]

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
                f"{churn_prob:.1%}" if pd.notna(churn_prob) else "N/A",
                f"{value_score:.2f}" if pd.notna(value_score) else "N/A",
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
    category_df = compute_category_counts(df, "customer_category", "Customer Category")
    actions_df = compute_category_counts(df, "recommended_action", "Recommended Action")
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
        sortable = df.copy()
        for col in sort_cols:
            sortable[col] = pd.to_numeric(sortable[col], errors="coerce")
        top_risk = sortable.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(10)[top_cols]

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
    prob_chart_df = compute_churn_distribution_by_actual(df)
    return stats, prob_chart_df


st.title("AI Customer Intelligence & Retention System")
st.caption("Streamlit version of your Gradio app, built for GitHub + Streamlit Community Cloud deployment.")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader(
        "Upload your processed project file",
        type=["csv", "xlsx", "xls"],
        help="If you do not upload a file, the app will try to read a project CSV from the repo.",
    )
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
churn_rate = float(pd.to_numeric(df["Churn"], errors="coerce").dropna().mean()) if "Churn" in df.columns else 0.0
high_risk = int((pd.to_numeric(df["churn_probability"], errors="coerce") >= HIGH_RISK).fillna(False).sum()) if "churn_probability" in df.columns else 0
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
    st.info("This app is locked to your Telco project outputs and can run directly from your GitHub repo on Streamlit Cloud.")

    left, right = st.columns(2)

    with left:
        st.markdown("**Customer Risk vs Value**")
        scatter_df = compute_scatter_data(df)
        if not scatter_df.empty:
            st.scatter_chart(scatter_df, x="churn_probability", y="value_score", use_container_width=True)
            st.caption(f"Reference thresholds: high risk = {HIGH_RISK:.2f}, high value = {HIGH_VALUE:.2f}")
        else:
            st.warning("This chart needs both 'churn_probability' and 'value_score' columns.")

    with right:
        st.markdown("**Churn Probability Distribution**")
        hist_df = compute_histogram_counts(df["churn_probability"]) if "churn_probability" in df.columns else pd.DataFrame()
        if not hist_df.empty:
            st.bar_chart(hist_df.set_index("bin_label")[["count"]], use_container_width=True)
            st.caption(f"Customers at or above {HIGH_RISK:.2f} are considered high risk.")
        else:
            st.warning("This chart needs a valid 'churn_probability' column.")

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.markdown("**Customer Category Distribution**")
        category_chart_df = compute_category_counts(df, "customer_category", "Customer Category")
        if not category_chart_df.empty:
            st.vega_lite_chart(
                category_chart_df,
                {
                    "mark": {"type": "arc", "innerRadius": 60},
                    "encoding": {
                        "theta": {"field": "Count", "type": "quantitative"},
                        "color": {
    "field": "Customer Category",
    "type": "nominal",
    "legend": {"title": "Customer Category"},
    "scale": {
        "domain": [
            "High Value - Low Risk",
            "High Value - High Risk",
            "Low Value - Low Risk",
            "Low Value - High Risk"
        ],
        "range": [
            "#2ECC71",  # green → best customers
            "#E74C3C",  # red → urgent risk
            "#3498DB",  # blue → stable but low value
            "#F39C12"   # orange → risky low value
        ]
    }
},
                        "tooltip": [
                            {"field": "Customer Category", "type": "nominal"},
                            {"field": "Count", "type": "quantitative"},
                        ],
                    },
                },
                use_container_width=True,
            )
        else:
            st.warning("This chart needs a 'customer_category' column.")

    with bottom_right:
        st.markdown("**Recommended Actions**")
        action_chart_df = compute_category_counts(df, "recommended_action", "Recommended Action")
        if not action_chart_df.empty:
            st.bar_chart(action_chart_df.set_index("Recommended Action")[["Count"]], use_container_width=True)
        else:
            st.warning("This chart needs a 'recommended_action' column.")

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
        st.dataframe(category_df, use_container_width=True, hide_index=True)
    with b2:
        st.markdown("**Recommended Actions Distribution**")
        st.dataframe(actions_df, use_container_width=True, hide_index=True)

    st.markdown("**Top Customers to Save**")
    st.dataframe(top_risk_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Model Insights")
    stats_df, prob_chart_df = get_model_insights(df)

    st.markdown("**Feature Snapshot**")
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("**Probability Distribution by Actual Churn**")
    if not prob_chart_df.empty:
        plot_df = prob_chart_df.set_index("probability_bin")
        st.bar_chart(plot_df, use_container_width=True)
    else:
        st.warning("Model insight chart needs both 'Churn' and 'churn_probability' columns.")

with tabs[4]:
    st.subheader("Export")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name="customer_intelligence_results.csv",
        mime="text/csv",
    )
