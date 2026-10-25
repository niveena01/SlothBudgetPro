# 🦥 SlothBudget Pro — Autonomous Offline Financial Agent

> **Built for the DEV x Hacktoberfest Open-Source AI Challenge ("Slow and Steady" Theme)**  
> *Zero-effort expense tracking and categorization powered by local open-weight models, tabular machine learning, and cloud dashboards.*

---

## 🌟 What It Is & The Lazy Payoff
Manually organizing bank statements, UPI transactions, and receipt text files is a tedious administrative chore. **SlothBudget Pro** completely automates financial housekeeping in the background. 

You simply drop raw receipt logs, SMS alerts, or text notes into a local folder (or upload them via the cloud dashboard), and the autonomous agent handles feature extraction, tabular ML classification, and ledger recording—**requiring zero manual data entry.**

---

## 🏗️ Hybrid Architecture (Local Edge + Render Cloud)

SlothBudget Pro utilizes a cutting-edge **hybrid architecture** optimized to run smoothly on standard consumer hardware (e.g., 8GB RAM Windows PCs):

1. **Frontend Runtime (Render Cloud):** Hosted live as an interactive Streamlit Web Service on **Render**, allowing global access and real-time visualization.
2. **Unstructured Text Extraction (Local Edge):** Google's open-weight **Gemma 3 (1B)** model extracts structured transaction details (merchant, amount, context) from raw text locally via Ollama CPU.
3. **Tabular Machine Learning Classification (Local Edge):** Prior Labs' **TabPFN-3.5** tabular foundation model classifies spending patterns (Food, Bills, Shopping, Travel) on extracted tabular features without cloud dependencies.
4. **Observability & Agent Tracing:** Integrated **Sentry Agent Tracing** tracks multi-model inference latency, spans, and error rates.

---

## 🚀 Key Features
- **100% Financial Privacy:** Sensitive banking and receipt records are processed locally on your hardware. No financial data ever leaks to third-party APIs.
- **Zero Cost:** Runs locally using open weights with $0 in API subscription fees.
- **Resilient Background Agent:** Built with Python `watchdog` to monitor filesystem events and execute automated tasks instantly.
- **Live Cloud Dashboard:** Real-time analytics, category breakdowns, and file upload capabilities hosted on Render.

---

## 📂 Project Structure

```text
SlothBudgetPro/
│
├── .streamlit/
│   └── config.toml           # Streamlit server configuration for Render
├── Unsorted/                 # Drop raw bank SMS / receipt text files here
├── Processed/                # Automatically archived raw files
├── app.py                    # Local background AI agent (Gemma 3 + TabPFN + Sentry)
├── dashboard.py              # Interactive Streamlit frontend dashboard (Render-ready)
├── requirements.txt          # Python dependencies
├── render.yaml               # Render cloud deployment configuration
└── ledger.csv                # Generated local financial ledger database