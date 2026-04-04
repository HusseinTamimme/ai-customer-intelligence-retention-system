# AI Customer Intelligence & Retention System

##  Overview

An end-to-end AI-powered system that predicts customer churn, segments users, and generates actionable retention strategies through an interactive dashboard.

This project transforms raw customer data into business intelligence, enabling companies to identify high-risk customers, prioritize retention efforts, and protect revenue.

---

##  Business Problem

Customer churn is a critical issue for subscription-based businesses.

This system answers key business questions:

* Which customers are most likely to churn?
* What drives customer churn?
* Which customers are most valuable?
* How can retention efforts be optimized?

---

##  Solution Architecture

This project follows a full data science pipeline:

1. **Data Understanding**
2. **Data Cleaning**
3. **Feature Engineering**
4. **Customer Segmentation**
5. **Churn Prediction (Machine Learning)**
6. **Customer Value Modeling**
7. **Decision Engine (Actionable Insights)**

---

##  Project Structure

###  Notebooks

* `01_data_understanding.ipynb`
* `02_data_cleaning.ipynb`
* `03_feature_engineering.ipynb`
* `04_customer_segmentation.ipynb`
* `05_churn_prediction.ipynb`
* `06_customer_value_model.ipynb`
* `07_decision_engine.ipynb`

###  Generated Data Files

* `cleaned_data.csv`
* `engineered_data.csv`
* `segmented_data.csv`
* `modeled_churn_data.csv`
* `final_scored_data.csv`
* `business_ready_data.csv`
* `customer_intelligence_results.csv`

### 🌐 Application

* `streamlit_app.py`
* `requirements.txt`
* `gradio_app.ipynb`  

---

## ⚙️ How the System Works

###  Data Pipeline

Raw dataset → cleaned → engineered → segmented → modeled → scored → decision-ready output

###  Final Output

The system produces a **business-ready dataset** containing:

* Churn probability
* Customer value score
* Customer segment/category
* Recommended retention action

This output feeds directly into the Streamlit application 

---

## Key Features

* End-to-end data pipeline
* Churn prediction model (ML)
* Customer segmentation (behavior + value)
* Customer value scoring system
* Revenue at risk estimation
* Decision engine with actionable recommendations
* Interactive dashboard (Streamlit)

---

##  Key Insights

* High-value customers can still have high churn risk → critical retention targets
* Month-to-month contracts show the highest churn probability
* Low engagement strongly correlates with churn
* A small percentage of customers drives a large portion of revenue at risk

---

##  Decision Engine (Core Innovation)

This system goes beyond prediction by **translating insights into actions**:

* Identify high-risk customers
* Recommend retention strategies
* Highlight high-value opportunities
* Prioritize customers based on risk vs value

This bridges the gap between **data science and business decision-making**.

---

##  Interactive Dashboard

The Streamlit app provides:

* Customer-level analysis (lookup system)
* Business overview (KPIs & distributions)
* Risk vs Value visualization
* Recommended actions dashboard
* Exportable results

The app automatically loads project datasets or allows manual upload 

---

##  Tech Stack

* Python (Pandas, NumPy)
* Scikit-learn
* Matplotlib
* Streamlit
* Jupyter Notebook

---

##  How to Run

1. Clone the repository

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run streamlit_app.py
```

---


## 🚀 Future Improvements

* Deploy as a full production system
* Integrate real-time data pipelines
* Improve model performance (XGBoost, tuning)
* Automate retention campaigns
* Add user authentication to dashboard

---

##  Author

Hussein (Nizar) Tamimme
Data Analyst | Aspiring Machine Learning Specialist
