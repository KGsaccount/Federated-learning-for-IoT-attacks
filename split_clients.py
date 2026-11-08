"""Split the full dataset into one CSV per federated client (stratified, IID).

Usage:
    python split_clients.py --data data/full_dataset.csv --clients 3
Creates data/client1_data.csv ... data/clientN_data.csv
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from preprocessing import TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    df = pd.read_csv(args.data, on_bad_lines="skip", low_memory=False)
    os.makedirs(args.out, exist_ok=True)
    folds = StratifiedKFold(n_splits=args.clients, shuffle=True, random_state=42)
    for i, (_, idx) in enumerate(folds.split(df, df[TARGET]), start=1):
        path = os.path.join(args.out, f"client{i}_data.csv")
        df.iloc[idx].to_csv(path, index=False)
        print(f"{path}: {len(idx)} rows")


if __name__ == "__main__":
    main()
