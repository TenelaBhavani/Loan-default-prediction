import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Default dataset filenames (must be in the same folder as this script)
    accepted_path = "accepted.csv"
    rejected_path = "rejected.csv"

    # Check if files exist
    if not os.path.exists(accepted_path) or not os.path.exists(rejected_path):
        print(f"Error: Could not find {accepted_path} or {rejected_path}")
        sys.exit(1)

    print("\nLoading datasets...")
    accepted = pd.read_csv(accepted_path, low_memory=False)
    rejected = pd.read_csv(rejected_path, low_memory=False)

    print(f"Accepted dataset shape: {accepted.shape}")
    print(f"Rejected dataset shape: {rejected.shape}")

    # ✅ Preprocessing (basic cleaning)
    if "loan_status" not in accepted.columns:
        print("Error: 'loan_status' column not found in accepted dataset.")
        sys.exit(1)

    accepted = accepted.dropna(axis=1, thresh=0.8*len(accepted))  # drop mostly empty cols
    accepted = accepted.dropna()  # drop rows with NaN

    # Convert categorical columns
    for col in accepted.select_dtypes(include=["object"]).columns:
        accepted[col] = accepted[col].astype("category").cat.codes

    # Features and target
    X = accepted.drop("loan_status", axis=1)
    y = accepted["loan_status"]

    # Train-test split
    print("\nSplitting dataset into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("\n✅ Loan default prediction completed!")

if __name__ == "__main__":
    main()
