
import os
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
XLSX_PATH = os.path.join(BASE_DIR, "data", "Gen_AI Dataset.xlsx")
EVAL_DIR  = os.path.join(BASE_DIR, "evaluation")

def extract():
    print(f"Reading: {XLSX_PATH}")

    # Train-Set sheet
    train = pd.read_excel(XLSX_PATH, sheet_name="Train-Set")
    train.columns = ["query", "assessment_url"]
    train_path = os.path.join(EVAL_DIR, "train.csv")
    train.to_csv(train_path, index=False, encoding="utf-8")
    print(f"Saved train.csv  — {len(train)} rows  → {train_path}")

    # Test-Set sheet
    test = pd.read_excel(XLSX_PATH, sheet_name="Test-Set")
    test.columns = ["query"]
    test_path = os.path.join(EVAL_DIR, "test.csv")
    test.to_csv(test_path, index=False, encoding="utf-8")
    print(f"Saved test.csv   — {len(test)} rows  → {test_path}")

    print("\nDone.")

if __name__ == "__main__":
    extract()
