import streamlit as st
import pandas as pd
import os
import time

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

# Sidebar Control Panel & Transaction Input (Upload + Paste)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/sloth.png", width=80)
    st.header("🦥 Agent Control Panel")
    st.success("🟢 Agent Status: Active (Local Engine)")
    st.info("☁️ Frontend Hosted on Render Cloud")
    
    st.subheader("📥 Add Transactions")
    upload_tab, paste_tab = st.tabs(["📁 Upload File", "📋 Paste Text"])
    
    with upload_tab:
        uploaded_file = st.file_uploader("Drop receipt or SMS text file here:", type=["txt", "csv"], key="file_uploader")
        if uploaded_file is not None:
            file_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_key") != file_key:
                save_path = os.path.join(UNSORTED_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state["last_uploaded_key"] = file_key
                st.success(f"Sent `{uploaded_file.name}` to Agent!")
            else:
                st.info(f"📁 `{uploaded_file.name}` ready for processing.")

    with paste_tab:
        if "pasted_text_val" not in st.session_state:
            st.session_state["pasted_text_val"] = ""

        col_sample, col_clear = st.columns(2)
        with col_sample:
            if st.button("💡 Load Sample", use_container_width=True, help="Insert sample receipt text"):
                st.session_state["pasted_text_val"] = "Paid $32.40 at Shell Gas Station on Sunday at 7 PM."
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["pasted_text_val"] = ""

        pasted_text = st.text_area(
            "Paste receipt or SMS alert:",
            value=st.session_state.get("pasted_text_val", ""),
            placeholder="e.g. Paid $14.50 to Starbucks on Tuesday at 9 AM.",
            height=110
        )
        submit_paste = st.button("📤 Send to Agent", use_container_width=True, key="submit_paste_btn")

        if submit_paste:
            clean_paste = pasted_text.strip()
            if clean_paste:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                paste_filename = f"pasted_receipt_{timestamp_str}.txt"
                save_path = os.path.join(UNSORTED_DIR, paste_filename)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(clean_paste)
                st.session_state["pasted_text_val"] = ""
                st.success(f"Sent `{paste_filename}` to Agent!")
                st.rerun()
            else:
                st.warning("Please paste transaction text before submitting.")

    # Check for pending files in Unsorted
    pending_files = [f for f in os.listdir(UNSORTED_DIR) if os.path.isfile(os.path.join(UNSORTED_DIR, f)) and f.endswith((".txt", ".csv"))]
    if pending_files:
        st.markdown("---")
        st.warning(f"⏳ **{len(pending_files)} pending file(s)** waiting in `Unsorted/`")
        if st.button("⚡ Process Pending Now", use_container_width=True, help="Run background extraction & categorization immediately"):
            with st.spinner("Processing unsorted files..."):
                try:
                    from app import SlothBudgetAgent
                    agent = SlothBudgetAgent()
                    for pf in sorted(pending_files):
                        agent.process_file(os.path.join(UNSORTED_DIR, pf))
                    st.success("Processed all pending files!")
                    st.rerun()
                except Exception as proc_err:
                    st.error(f"Processing error: {proc_err}")

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

if os.path.exists(DATABASE_FILE) and os.path.getsize(DATABASE_FILE) > 0:
    try:
        df = pd.read_csv(DATABASE_FILE)
        if not df.empty and "category" in df.columns:
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
                    st.write(f"• **{row['merchant']}** — `${row['amount']:.2f}` ({row['category']}) on *{row['timestamp']}*")
        else:
            st.info("No transactions logged yet. Drop an SMS or receipt text file into the sidebar uploader or your local `Unsorted/` folder!")
    except Exception as e:
        st.info("No transactions logged yet. Drop an SMS or receipt text file into the sidebar uploader or your local `Unsorted/` folder!")
else:
    st.info("No transactions logged yet. Drop an SMS or receipt text file into the sidebar uploader or your local `Unsorted/` folder!")