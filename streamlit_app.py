
from pathlib import Path
import matplotlib.pyplot as plt
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
    return float(df[column].mean()) if column in df.columns else default


def metric_sum(df, column, mask=None, default=0):
    if column not in df.columns:
        return default
    if mask is None:
        return float(df[column].sum())
    return float(df.loc[mask, column].sum())


def build_overview_figures(df: pd.DataFrame):
    figures = []

    if {"churn_probability", "value_score"}.issubset(df.columns):
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(df["churn_probability"], df["value_score"], alpha=0.5)
        ax1.axvline(HIGH_RISK, linestyle="--")
        ax1.axhline(HIGH_VALUE, linestyle="--")
        ax1.set_xlabel("Churn Probability")
        ax1.set_ylabel("Value Score")
        ax1.set_title("Customer Risk vs Value")
        figures.append(("Customer Risk vs Value", fig1))

    if "customer_category" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df["customer_category"].value_counts().plot(kind="barh", ax=ax2)
        ax2.set_xlabel("Customers")
        ax2.set_ylabel("")
        ax2.set_title("Customer Category Distribution")
        figures.append(("Customer Category Distribution", fig2))

    if "recommended_action" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        df["recommended_action"].value_counts().plot(kind="barh", ax=ax3)
        ax3.set_xlabel("Customers")
        ax3.set_ylabel("")
        ax3.set_title("Recommended Actions")
        figures.append(("Recommended Actions", fig3))

    if "churn_probability" in df.columns:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.hist(df["churn_probability"], bins=25, alpha=0.8)
        ax4.axvline(HIGH_RISK, linestyle="--")
        ax4.set_xlabel("Churn Probability")
        ax4.set_ylabel("Count")
        ax4.set_title("Churn Probability Distribution")
        figures.append(("Churn Probability Distribution", fig4))

    return figures


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
    category_df = pd.DataFrame()
    actions_df = pd.DataFrame()
    top_risk = pd.DataFrame()

    if "customer_category" in df.columns:
        category_df = df["customer_category"].value_counts().reset_index()
        category_df.columns = ["Customer Category", "Count"]

    if "recommended_action" in df.columns:
        actions_df = df["recommended_action"].value_counts().reset_index()
        actions_df.columns = ["Recommended Action", "Count"]

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

    fig = None
    if {"Churn", "churn_probability"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df.loc[df["Churn"] == 0, "churn_probability"], bins=25, alpha=0.6, label="Stayed")
        ax.hist(df.loc[df["Churn"] == 1, "churn_probability"], bins=25, alpha=0.6, label="Churned")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Count")
        ax.set_title("Probability Distribution by Actual Churn")
        ax.legend()

    return stats, fig


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
churn_rate = float(df["Churn"].mean()) if "Churn" in df.columns else 0.0
high_risk = int((df["churn_probability"] >= HIGH_RISK).sum()) if "churn_probability" in df.columns else 0
avg_value = metric_value(df, "value_score", 0.0)
revenue_at_risk = (
    metric_sum(df, "Total Charges", df["churn_probability"] >= HIGH_RISK, 0.0)
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

    figures = build_overview_figures(df)
    if figures:
        for i in range(0, len(figures), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(figures):
                    title, fig = figures[i + j]
                    with col:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
    else:
        st.warning("The required overview columns were not found in the dataset.")

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
    stats_df, prob_fig = get_model_insights(df)

    st.markdown("**Feature Snapshot**")
    st.dataframe(stats_df, use_container_width=True)

    if prob_fig is not None:
        st.pyplot(prob_fig, use_container_width=True)
        plt.close(prob_fig)
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
