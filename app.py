import streamlit as st
import pandas as pd
import plotly.express as px
from models.isolation_forest import run_isolation_forest
from models.autoencoder import run_autoencoder
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

st.set_page_config(page_title="AI-Powered Anomaly Detection", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f8;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #003566;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4f4f4f;
    }
    .stButton > button {
        background-color: #003566;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🔍 Select Model")
model_choice = st.sidebar.radio("Choose an algorithm:", ["Isolation Forest", "Autoencoder (PyTorch)"])

# ---------- Title Section ----------
st.markdown("<div class='title'>💸 AI-Powered Anomaly Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Identify fraudulent or atypical patterns in financial transaction data using machine learning models.</div>", unsafe_allow_html=True)

# ---------- Upload or Load Default ----------
st.markdown("### 📤 Upload your transaction data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
default_file_path = 'sample.csv'

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.markdown("### 📊 Preview of Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
elif os.path.exists(default_file_path):
    df = pd.read_csv(default_file_path)
    st.success("✅ Default dataset loaded.")
    st.markdown("### 📊 Preview of Default Dataset")
    st.dataframe(df.head(), use_container_width=True)
else:
    st.warning("📂 Please upload a CSV file or ensure 'sample.csv' exists.")

# ---------- Anomaly Detection ----------
if df is not None:
    st.markdown("### ⚙️ Select Features for Anomaly Detection")
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    selected_features = st.multiselect("Choose numeric columns", options=numeric_columns)

    if selected_features:
        st.markdown(f"#### Model Selected: **{model_choice}**")
        if st.button("🚀 Run Detection"):
            with st.spinner("Running anomaly detection..."):
                true_labels = None
                if 'true_label' in df.columns:
                    true_labels = df['true_label']

                if model_choice == "Isolation Forest":
                    df, anomaly_scores, metrics = run_isolation_forest(df[selected_features], true_labels)
                else:
                    df, anomaly_scores, metrics = run_autoencoder(df[selected_features], true_labels)

                df['anomaly_score'] = anomaly_scores

                # ---------- Visualization 1: Scatter Plot ----------
                st.markdown("### 📈 Anomaly Scatter Plot")
                fig = px.scatter(
                    df,
                    x=selected_features[0],
                    y=selected_features[1] if len(selected_features) > 1 else selected_features[0],
                    color=df['anomaly_score'].map({0: 'Normal', 1: 'Anomaly'}),
                    title="Transaction Anomalies"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---------- Visualization 2: Anomaly Count ----------
                st.markdown("### 📊 Anomaly Count Summary")
                count_df = df['anomaly_score'].value_counts().rename(index={0: 'Normal', 1: 'Anomaly'})
                fig_bar = px.bar(x=count_df.index, y=count_df.values, color=count_df.index,
                                 labels={'x': 'Label', 'y': 'Count'}, title="Anomalies vs. Normal Transactions")
                st.plotly_chart(fig_bar, use_container_width=True)

                # ---------- Visualization 3: Time Series (if applicable) ----------
                if 'transaction_date' in df.columns:
                    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
                    time_df = df.dropna(subset=['transaction_date'])
                    st.markdown("### 📉 Time Series Visualization")
                    fig_time = px.line(
                        time_df,
                        x='transaction_date',
                        y=selected_features[0],
                        color=time_df['anomaly_score'].map({0: 'Normal', 1: 'Anomaly'}),
                        title=f"{selected_features[0]} Over Time with Anomalies"
                    )
                    st.plotly_chart(fig_time, use_container_width=True)

                # ---------- Anomaly Table ----------
                st.markdown("### 🧾 Detailed Anomaly Report")
                anomaly_df = df[df['anomaly_score'] == 1]
                if not anomaly_df.empty:
                    st.dataframe(anomaly_df, use_container_width=True)
                else:
                    st.info("No anomalies detected in the selected features.")

                # ---------- Performance Metrics ----------
                if metrics:
                    st.markdown("### 📊 Model Performance Metrics")
                    st.write(f"**Precision**: {metrics['precision']:.2f}")
                    st.write(f"**Recall**: {metrics['recall']:.2f}")
                    st.write(f"**F1 Score**: {metrics['f1_score']:.2f}")
                    st.write(f"**Accuracy**: {metrics['accuracy']:.2f}")

                # ---------- Closing Summary ----------
                st.markdown("### ✅ Report Summary")
                st.success("""
                ✅ The anomaly detection process has successfully completed.

                🔍 What we did:
                - Analyzed the selected numerical features using the chosen model.
                - Flagged all potential anomalies.
                - Displayed visual reports and detailed tables for better understanding.

                📊 You can try different features or upload another dataset to explore further insights.
                """)

