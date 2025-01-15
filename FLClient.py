import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import load

def load_client_data(file_path, label_encoder=None):
    """
    Load and preprocess data for a specific client.
    Args:
        file_path (str): Path to the client's dataset.
        label_encoder (LabelEncoder, optional): Encoder for consistent label transformation.
    Returns:
        tuple: Preprocessed features (X), target (y), and the label encoder.
    """
    df = pd.read_csv(file_path, on_bad_lines="skip")
    irrelevant_columns = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]
    df = df.drop(columns=irrelevant_columns, errors="ignore")

    if df.isnull().values.any():
        df.fillna(df.mean(), inplace=True)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["Attack_Name_Encoded"] = label_encoder.fit_transform(df["Attack Name"])
    else:
        df["Attack_Name_Encoded"] = label_encoder.transform(df["Attack Name"])

    X = df.drop(columns=["Attack Name", "Attack_Name_Encoded"])
    y = df["Attack_Name_Encoded"]
    return X, y, label_encoder

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_data_path, initial_model, noise_level, decay_rate):
        self.client_data_path = client_data_path
        self.initial_model = initial_model
        self.noise_level = noise_level
        self.decay_rate = decay_rate
        self.round_num = 0
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        """
        Load and preprocess client data.
        """
        X, y, _ = load_client_data(self.client_data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def get_parameters(self):
        """
        Extract parameters from the local model.
        """
        return [self.initial_model.coef_, self.initial_model.intercept_]

    def set_parameters(self, parameters):
        """
        Set parameters to the local model.
        """
        self.initial_model.coef_, self.initial_model.intercept_ = parameters

    def fit(self, parameters, config):
        """
        Perform local training with adaptive Gaussian noise.
        """
        self.set_parameters(parameters)
        self.initial_model.fit(self.X_train, self.y_train)

        # Apply adaptive Gaussian noise
        adaptive_noise_level = self.noise_level * np.exp(-self.decay_rate * self.round_num)
        self.initial_model.coef_ += np.random.normal(0, adaptive_noise_level, self.initial_model.coef_.shape)

        self.round_num += 1
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the local model.
        """
        self.set_parameters(parameters)
        predictions = self.initial_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return len(self.X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    client_id = int(input("Enter client ID (1-9): "))
    client_data_path = f"client{client_id}_data.csv"
    initial_model_path = "path/to/initial_model.joblib"  # Update with the actual path
    initial_model = load(initial_model_path)
    print(f"Starting client {client_id} with data: {client_data_path}")

    noise_level = 0.5
    decay_rate = 0.1

    # Start the Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_data_path, initial_model, noise_level, decay_rate)
    )
