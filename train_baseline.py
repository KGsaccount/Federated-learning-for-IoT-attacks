"""Train a centralised baseline model on the full dataset.

The SGD model trained here is also the starting point of federated training:
it fixes the feature list, the scaler and the list of attack classes that all
clients share.

Usage:
    python train_baseline.py --data data/full_dataset.csv --model sgd
    python train_baseline.py --data data/full_dataset.csv --model naive_bayes
    python train_baseline.py --data data/full_dataset.csv --model hoeffding
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocessing import clean_features, load_dataset

MODELS_DIR = "trained_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with an 'Attack Name' column")
    parser.add_argument("--model", choices=["sgd", "naive_bayes", "hoeffding"], default="sgd")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    print(f"Loaded {len(X)} rows, {X.shape[1]} numeric features, classes: {sorted(y.unique())}")
    if y.nunique() < 2:
        raise SystemExit("The dataset needs at least two classes (benign + attacks) to train a classifier.")

    encoder = LabelEncoder().fit(y)
    y_enc = encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    columns = list(X.columns)
    fill_values = X_train.median()
    scaler = StandardScaler().fit(clean_features(X_train, columns, fill_values))
    X_train = scaler.transform(clean_features(X_train, columns, fill_values))
    X_test = scaler.transform(clean_features(X_test, columns, fill_values))

    if args.model == "sgd":
        model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.model == "naive_bayes":
        model = GaussianNB().fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        from river.tree import HoeffdingTreeClassifier  # streaming model, trained one sample at a time

        model = HoeffdingTreeClassifier()
        for row, label in zip(X_train, y_train):
            model.learn_one(dict(enumerate(row)), int(label))
        y_pred = np.array([model.predict_one(dict(enumerate(row))) for row in X_test], dtype=float)
        y_pred = np.nan_to_num(y_pred, nan=-1).astype(int)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=range(len(encoder.classes_)),
                                target_names=encoder.classes_, zero_division=0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, f"{MODELS_DIR}/{args.model}_model.joblib")
    if args.model == "sgd":
        # Shared preprocessing for the federated clients.
        joblib.dump({"columns": columns, "fill_values": fill_values, "scaler": scaler, "encoder": encoder},
                    f"{MODELS_DIR}/preprocessor.joblib")
    print(f"Saved to {MODELS_DIR}/{args.model}_model.joblib")


if __name__ == "__main__":
    main()
