import os
import sys
import time
import shutil
import json
import re
import tabpfn_client

# Ensure standard output can handle UTF-8 symbols and emojis on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import ollama
import sentry_sdk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from plyer import notification
from tabpfn import TabPFNClassifier

# Initialize Sentry Agent Tracing (Sponsor Category Prize)
# sentry_sdk.init(
#     dsn="YOUR_SENTRY_DSN_HERE",  # Replace with your free Sentry DSN from sentry.io
#     traces_sample_rate=1.0,
# )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCH_DIR = os.path.join(BASE_DIR, "Unsorted")
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed")
DATABASE_FILE = os.path.join(BASE_DIR, "ledger.csv")

os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Read from Render Environment Variables (or fallback to local setting)
tabpfn_key = os.getenv("TABPFN_API_KEY") or os.getenv("TABPFN_TOKEN")

if tabpfn_key:
    tabpfn_client.set_access_token(tabpfn_key)

# Initialize and fit TabPFN-3.5 foundation model with seed dataset
def get_fitted_tabpfn():
    try:
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
    except Exception as e:
        print(f"⚠️ [Notice] TabPFN could not be initialized ({e}). Using rule-based fallback classification.")
        return None

tabpfn_model = get_fitted_tabpfn()
CATEGORY_MAP = {0: "Food", 1: "Shopping", 2: "Bills", 3: "Travel"}


def extract_details_fallback(raw_text: str) -> dict:
    """Fallback extraction directly from raw text content when LLM fails or errors."""
    clean_text = raw_text.strip()

    # 1. Amount extraction
    amount = 0.0
    amt_match = re.search(r'(?:[\$€£₹]|(?:USD|EUR|GBP|INR|Rs\.?)\s*)([0-9]+(?:\.[0-9]{1,2})?)', clean_text, re.IGNORECASE)
    if not amt_match:
        amt_match = re.search(r'(?:spent|paid|amount|total|charge[d]?|debited|cost|fee)\s*(?:of\s*)?[\$€£₹]?\s*([0-9]+(?:\.[0-9]{1,2})?)', clean_text, re.IGNORECASE)
    if not amt_match:
        amt_match = re.search(r'\b([0-9]+\.[0-9]{2})\b', clean_text)
    if amt_match:
        try:
            amount = float(amt_match.group(1))
        except ValueError:
            amount = 0.0

    # 2. Merchant extraction: try prepositions or fallback directly to raw content
    merchant = ""
    merch_match = re.search(
        r'(?:at|to|from|for|in)\s+([A-Za-z0-9\s&\'\.-]+?)(?:\s+(?:\b(?:on|at|for|dated)\b|[\$€£₹]|\.|\n|$))',
        clean_text,
        re.IGNORECASE,
    )
    if merch_match:
        candidate = merch_match.group(1).strip()
        if not re.match(r'^[\$€£₹0-9\.]+$', candidate) and len(candidate) > 1:
            merchant = candidate

    # Fallback to the raw content itself if no merchant pattern matched
    if not merchant:
        first_line = clean_text.splitlines()[0] if clean_text else "Unknown"
        merchant = first_line[:80].strip() or "Unknown"

    # 3. Hour extraction
    hour = 12
    hour_match = re.search(r'\b([0-1]?[0-9]|2[0-3])(?::([0-5][0-9]))?\s*(am|pm)\b', clean_text, re.IGNORECASE)
    if hour_match:
        h = int(hour_match.group(1))
        meridiem = hour_match.group(3).lower()
        if meridiem == "pm" and h < 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        hour = h
    else:
        at_match = re.search(r'\bat\s*([0-2]?[0-9])\b', clean_text, re.IGNORECASE)
        if at_match:
            try:
                h = int(at_match.group(1))
                if 0 <= h <= 23:
                    hour = h
            except ValueError:
                pass

    # 4. Weekend extraction
    is_weekend = 0
    if re.search(r'\b(saturday|sunday|weekend)\b', clean_text, re.IGNORECASE):
        is_weekend = 1
    elif re.search(r'\b(monday|tuesday|wednesday|thursday|friday|weekday)\b', clean_text, re.IGNORECASE):
        is_weekend = 0

    return {
        "merchant": merchant,
        "amount": amount,
        "hour": hour,
        "is_weekend": is_weekend
    }


def infer_category_from_text(raw_text: str, merchant: str = "", amount: float = 0.0) -> str:
    """Infer financial category from text content when model inference is unavailable."""
    combined = f"{raw_text} {merchant}".lower()
    if any(k in combined for k in ["starbucks", "cafe", "coffee", "restaurant", "food", "burger", "pizza", "mcdonald", "dinner", "lunch", "breakfast", "meal", "grocery", "bakery", "subway", "zomato", "swiggy", "eats"]):
        return "Food"
    if any(k in combined for k in ["zara", "amazon", "walmart", "target", "store", "shop", "clothes", "clothing", "nike", "adidas", "shopping", "mall", "retail", "h&m"]):
        return "Shopping"
    if any(k in combined for k in ["bill", "utility", "electric", "water", "gas", "internet", "wifi", "rent", "recharge", "subscription", "insurance", "fee", "tax"]):
        return "Bills"
    if any(k in combined for k in ["uber", "lyft", "flight", "airline", "train", "railway", "metro", "bus", "fuel", "petrol", "diesel", "parking", "toll", "travel", "taxi", "trip"]):
        return "Travel"
    return "General"


class SlothBudgetAgent(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith((".txt", ".csv")):
            return
        
        time.sleep(1.0)
        self.process_file(event.src_path)

    def process_file(self, file_path):
        if not os.path.exists(file_path):
            return

        file_name = os.path.basename(file_path)
        print(f"\n🦥 [SlothAgent Triggered] Reading file: {file_name}")

        with sentry_sdk.start_transaction(op="agent_execution", name=f"Process_{file_name}"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except Exception as read_err:
                print(f"❌ Failed to read {file_name}: {read_err}")
                return

            parsed = None
            llm_latency = 0.0

            # Layer 1: Extract features via Gemma 3 1B (Local CPU) with Error Fallback to Content
            try:
                prompt = f"""
                Extract transaction details from text into valid JSON with keys: 'merchant', 'amount', 'hour', 'is_weekend'.
                Raw Text: {raw_text}
                Respond ONLY with valid JSON:
                """
                
                start_llm = time.time()
                response = ollama.generate(model="gemma3:1b", prompt=prompt)
                llm_latency = time.time() - start_llm
                sentry_sdk.set_measurement("gemma_llm_latency", llm_latency)
                
                clean_response = response.get("response", "").strip() if isinstance(response, dict) else getattr(response, "response", "").strip()
                if not clean_response and isinstance(response, dict):
                    clean_response = response.get("message", {}).get("content", "").strip()
                    
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(clean_response)
                if not isinstance(parsed, dict) or "amount" not in parsed:
                    raise ValueError("JSON response missing required transaction keys")
            except Exception as e:
                print(f"⚠️ [Layer 1 Fallback] LLM extraction failed ({e}). Falling back directly to file content.")
                sentry_sdk.capture_message(f"LLM extraction fallback used for {file_name}: {e}", level="warning")
                parsed = extract_details_fallback(raw_text)

            amount = float(parsed.get("amount", 0.0))
            hour = int(parsed.get("hour", 12))
            is_weekend = int(parsed.get("is_weekend", 0))
            merchant = parsed.get("merchant") or raw_text.strip().replace("\n", " ")[:80] or "Unknown"

            # Layer 2: Classify category using TabPFN-3.5 with fallback
            category = "General"
            tabpfn_latency = 0.0
            if tabpfn_model is not None:
                try:
                    start_tabpfn = time.time()
                    X_test = pd.DataFrame([{"amount": amount, "hour": hour, "is_weekend": is_weekend}])
                    pred_cat_id = tabpfn_model.predict(X_test)[0]
                    category = CATEGORY_MAP.get(pred_cat_id, "General")
                    tabpfn_latency = time.time() - start_tabpfn
                    sentry_sdk.set_measurement("tabpfn_inference_latency", tabpfn_latency)
                except Exception as tabpfn_err:
                    print(f"⚠️ [Layer 2 Fallback] TabPFN inference failed ({tabpfn_err}). Inferring category from content.")
                    category = infer_category_from_text(raw_text, merchant, amount)
            else:
                category = infer_category_from_text(raw_text, merchant, amount)

            # Layer 3: Save to local ledger database
            new_row = pd.DataFrame([{
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "merchant": merchant,
                "amount": amount,
                "category": category,
                "file": file_name
            }])
            
            try:
                if not os.path.exists(DATABASE_FILE) or os.path.getsize(DATABASE_FILE) == 0:
                    new_row.to_csv(DATABASE_FILE, index=False)
                else:
                    new_row.to_csv(DATABASE_FILE, mode='a', header=False, index=False)

                # Move raw file to processed folder
                shutil.move(file_path, os.path.join(PROCESSED_DIR, file_name))
                
                print(f"✅ Processed {file_name} -> {category} (${amount:.2f}) [Gemma: {llm_latency:.2f}s | TabPFN: {tabpfn_latency:.2f}s]")
                
                try:
                    notification.notify(
                        title="🦥 SlothBudget Pro",
                        message=f"Categorized ${amount:.2f} at {merchant} as {category}!",
                        timeout=3
                    )
                except Exception:
                    pass
            except Exception as save_err:
                sentry_sdk.capture_exception(save_err)
                print(f"❌ Error saving transaction for {file_name}: {save_err}")

if __name__ == "__main__":
    agent = SlothBudgetAgent()

    # Process any existing files in Unsorted folder on launch
    for existing_file in sorted(os.listdir(WATCH_DIR)):
        existing_path = os.path.join(WATCH_DIR, existing_file)
        if os.path.isfile(existing_path) and existing_file.endswith((".txt", ".csv")):
            agent.process_file(existing_path)

    observer = Observer()
    observer.schedule(agent, WATCH_DIR, recursive=False)
    observer.start()
    print("="*70)
    print("🦥 SLOTHBUDGET PRO ACTIVE (Gemma 3 1B + TabPFN-3.5 + Content Fallback)")
    print(f"📁 Drop bank SMS/receipt text files in: {WATCH_DIR}")
    print("="*70)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()