"""Flower client: trains on one site's local data, shares only model weights.

Usage:
    python client.py --id 1 --data data/client1_data.csv
"""
import argparse

import flwr as fl
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from preprocessing import clean_features, load_dataset


class IoTAttackClient(fl.client.NumPyClient):
    def __init__(self, data_path, noise_level, decay_rate, local_epochs):
        prep = joblib.load("trained_models/preprocessor.joblib")
        self.classes = np.arange(len(prep["encoder"].classes_))

        X, y = load_dataset(data_path)
        known = y.isin(prep["encoder"].classes_)  # ignore labels the global model doesn't know
        X = prep["scaler"].transform(clean_features(X[known], prep["columns"], prep["fill_values"]))
        y = prep["encoder"].transform(y[known])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Same model type as the centralised baseline; its weights always come from the server.
        self.model = SGDClassifier(loss="log_loss", penalty="l2", random_state=42)
        self.model.classes_ = self.classes
        self.noise_level, self.decay_rate, self.local_epochs = noise_level, decay_rate, local_epochs
        self.round_num = 0

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_, self.model.intercept_ = parameters

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(self.local_epochs):
            # partial_fit continues from the global weights; 'classes' keeps the shape identical on every client.
            self.model.partial_fit(self.X_train, self.y_train, classes=self.classes)

        # Gaussian noise on the shared weights, decreasing each round (noise-based privacy, not formal DP).
        sigma = self.noise_level * np.exp(-self.decay_rate * self.round_num)
        self.model.coef_ = self.model.coef_ + np.random.normal(0, sigma, self.model.coef_.shape)
        self.round_num += 1

        acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
        return self.get_parameters(config), len(self.X_train), {"accuracy": acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        proba = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, proba, labels=self.classes)
        acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
        return float(loss), len(self.X_test), {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--data", help="default: data/client<ID>_data.csv")
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--noise", type=float, default=0.05, help="initial noise standard deviation")
    parser.add_argument("--decay", type=float, default=0.1, help="noise decay rate per round")
    parser.add_argument("--epochs", type=int, default=1, help="local passes over the data per round")
    args = parser.parse_args()

    data_path = args.data or f"data/client{args.id}_data.csv"
    print(f"Client {args.id} using {data_path}")
    client = IoTAttackClient(data_path, args.noise, args.decay, args.epochs)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
