"""Flower server: coordinates federated training with FedAvg.

Usage:
    python server.py --rounds 10 --clients 3
"""
import argparse

import flwr as fl
import joblib
import numpy as np
from flwr.common import ndarrays_to_parameters


def weighted_average(metrics):
    """Average client accuracies, weighted by each client's number of test samples."""
    total = sum(num for num, _ in metrics)
    return {"accuracy": sum(num * m["accuracy"] for num, m in metrics) / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=3, help="clients required per round")
    parser.add_argument("--address", default="0.0.0.0:8080")
    args = parser.parse_args()

    # The global model starts from zero weights, so it learns only from the clients' data.
    # (Starting from the centralised baseline would leak the clients' test data into the model.)
    prep = joblib.load("trained_models/preprocessor.joblib")
    n_classes, n_features = len(prep["encoder"].classes_), len(prep["columns"])
    n_outputs = 1 if n_classes == 2 else n_classes
    initial = [np.zeros((n_outputs, n_features)), np.zeros(n_outputs)]

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=ndarrays_to_parameters(initial),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    print("\nFederated accuracy per round:")
    for rnd, acc in history.metrics_distributed.get("accuracy", []):
        print(f"  round {rnd}: {acc:.4f}")


if __name__ == "__main__":
    main()
