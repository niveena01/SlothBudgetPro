import os
import time
import shutil
import json
import pandas as pd
import numpy as np
import ollama
import sentry_sdk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from plyer import notification
from tabpfn import TabPFNClassifier

# Initialize Sentry Agent Tracing (Sponsor Category Prize)
sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN_HERE",  # Replace with your free Sentry DSN from sentry.io
    traces_sample_rate=1.0,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCH_DIR = os.path.join(BASE_DIR, "Unsorted")
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed")
DATABASE_FILE = os.path.join(BASE_DIR, "ledger.csv")

os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize and fit TabPFN-3.5 foundation model with seed dataset
def get_fitted_tabpfn():
    train_data = pd.DataFrame([
        {"amount": 15.50, "hour": 12, "is_weekend": 0, "cat": 0}, # Food
        {"amount": 120.00, "hour": 18, "is_weekend": 1, "cat": 1}, # Shopping
        {"amount": 45.00, "hour": 9, "is_weekend": 0, "cat": 2}, # Bills
        {"amount": 8.00, "hour": 8, "is_weekend": 0, "cat": 3},  # Travel
        {"amount": 25.00, "hour": 20, "is_weekend": 1, "cat": 0}, # Food
        {"amount": 300.00, "hour": 14, "is_weekend": 0, "cat": 1}, # Shopping
    ])
    X_train = train_data[["amount", "hour", "is_weekend"]]
    y_train = train_data["cat"]
    
    classifier = TabPFNClassifier()
    classifier.fit(X_train, y_train)
    return classifier

tabpfn_model = get_fitted_tabpfn()
CATEGORY_MAP = {0: "Food", 1: "Shopping", 2: "Bills", 3: "Travel"}

class SlothBudgetAgent(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith((".txt", ".csv")):
            return
        
        time.sleep(1.0)
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        print(f"\n🦥 [SlothAgent Triggered] Reading file: {file_name}")
        
        with sentry_sdk.start_transaction(op="agent_execution", name=f"Process_{file_name}"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                # Layer 1: Extract features via Gemma 3 1B (Local CPU)
                prompt = f"""
                Extract transaction details from text into valid JSON with keys: 'merchant', 'amount', 'hour', 'is_weekend'.
                Raw Text: {raw_text}
                Respond ONLY with valid JSON:
                """
                
                start_llm = time.time()
                response = ollama.generate(model="gemma3:1b", prompt=prompt)
                llm_latency = time.time() - start_llm
                sentry_sdk.set_measurement("gemma_llm_latency", llm_latency)
                
                # Parse JSON output safely
                clean_response = response["response"].strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(clean_response)
                amount = float(parsed.get("amount", 0.0))
                hour = int(parsed.get("hour", 12))
                is_weekend = int(parsed.get("is_weekend", 0))
                merchant = parsed.get("merchant", "Unknown")

                # Layer 2: Classify category using TabPFN-3.5
                start_tabpfn = time.time()
                X_test = pd.DataFrame([{"amount": amount, "hour": hour, "is_weekend": is_weekend}])
                pred_cat_id = tabpfn_model.predict(X_test)[0]
                category = CATEGORY_MAP.get(pred_cat_id, "General")
                tabpfn_latency = time.time() - start_tabpfn
                sentry_sdk.set_measurement("tabpfn_inference_latency", tabpfn_latency)

                # Layer 3: Save to local ledger database
                new_row = pd.DataFrame([{
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "merchant": merchant,
                    "amount": amount,
                    "category": category,
                    "file": file_name
                }])
                
                if not os.path.exists(DATABASE_FILE):
                    new_row.to_csv(DATABASE_FILE, index=False)
                else:
                    new_row.to_csv(DATABASE_FILE, mode='a', header=False, index=False)

                # Move raw file to processed folder
                shutil.move(file_path, os.path.join(PROCESSED_DIR, file_name))
                
                print(f"✅ Processed {file_name} -> {category} (${amount}) [Gemma: {llm_latency:.2f}s | TabPFN: {tabpfn_latency:.2f}s]")
                
                notification.notify(
                    title="🦥 SlothBudget Pro",
                    message=f"Categorized ${amount} at {merchant} as {category}!",
                    timeout=3
                )
                
            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"❌ Error processing transaction: {e}")

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(SlothBudgetAgent(), WATCH_DIR, recursive=False)
    observer.start()
    print("="*70)
    print("🦥 SLOTHBUDGET PRO ACTIVE (Gemma 3 1B + TabPFN-3.5 + Sentry Tracing)")
    print(f"📁 Drop bank SMS/receipt text files in: {WATCH_DIR}")
    print("="*70)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()