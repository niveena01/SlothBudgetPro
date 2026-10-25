import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="SlothBudget Pro — Autonomous AI Financial Agent",
    page_icon="🦥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

st.title("🦥 SlothBudget Pro: Autonomous Offline Financial Agent")
st.caption("Hosted on **Render Cloud** • Powered by **Google Gemma 3 (1B)**, **Prior Labs TabPFN-3.5**, & **Sentry Tracing**")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, "ledger.csv")
UNSORTED_DIR = os.path.join(BASE_DIR, "Unsorted")
os.makedirs(UNSORTED_DIR, exist_ok=True)

# Sidebar Control Panel & File Uploader
with st.sidebar:
    st.image("https://img.icons8.com/color/96/sloth.png", width=80)
    st.header("🦥 Agent Control Panel")
    st.success("🟢 Agent Status: Active (Local Engine)")
    st.info("☁️ Frontend Hosted on Render Cloud")
    
    st.subheader("📤 Quick Upload File")
    uploaded_file = st.file_uploader("Drop receipt or SMS text file here:", type=["txt", "csv"])
    if uploaded_file is not None:
        save_path = os.path.join(UNSORTED_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Sent `{uploaded_file.name}` to Local Agent!")

# KPI Metrics Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Web Runtime", "Render Web Service", "Live Cloud")
with col2:
    st.metric("Text Model", "Gemma 3 1B", "Local CPU Inference")
with col3:
    st.metric("Tabular ML", "TabPFN-3.5", "Prior Labs Foundation")
with col4:
    st.metric("Privacy Level", "100% Offline Engine", "Zero Data Leaks")

st.markdown("---")
st.subheader("📊 Live Categorized Financial Ledger")

if os.path.exists(DATABASE_FILE):
    df = pd.read_csv(DATABASE_FILE)
    
    cat_filter = st.multiselect("Filter by Category:", options=df["category"].unique(), default=df["category"].unique())
    filtered_df = df[df["category"].isin(cat_filter)]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Spending Breakdown")
        cat_summary = df.groupby("category")["amount"].sum().reset_index()
        st.bar_chart(cat_summary, x="category", y="amount")
    
    with col_b:
        st.subheader("🔍 Recent Transactions")
        for idx, row in df.tail(5).iterrows():
            st.write(f"• **{row['merchant']}** — `${row['amount']}` ({row['category']}) on *{row['timestamp']}*")
else:
    st.info("No transactions logged yet. Drop an SMS or receipt text file into the sidebar uploader or your local `Unsorted/` folder!")