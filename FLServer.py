import flwr as fl
import numpy as np

def weighted_average(metrics):
    """
    Compute weighted average of metrics (e.g., loss, accuracy).
    Args:
        metrics (list): List of tuples containing (number_of_examples, metric_value).
    Returns:
        dict: Dictionary with the aggregated metric.
    """
    num_examples = [num for num, _ in metrics]
    accuracies = [acc for _, acc in metrics]
    total_examples = sum(num_examples)
    avg_accuracy = sum(num * acc for num, acc in zip(num_examples, accuracies)) / total_examples
    return {"accuracy": avg_accuracy}

# Define the Flower server strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

if __name__ == "__main__":
    # Start the server
    print("Starting Flower server...")
    fl.server.start_server(server_address="0.0.0.0:8080", config={"num_rounds": 10}, strategy=strategy)
