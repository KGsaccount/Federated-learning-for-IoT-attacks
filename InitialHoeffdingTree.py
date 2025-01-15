# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from river.tree import HoeffdingTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from datetime import datetime

def train_hoeffding_tree_model():
    # Loading the dataset
    file_path = "C:/datasets/Benign_Traffic.csv"  
    df = pd.read_csv(file_path, on_bad_lines='skip')

    print("Dataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

    # Dropping irrelevant columns
    irrelevant_columns = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']
    df = df.drop(columns=irrelevant_columns, errors='ignore')

    # Handling missing values
    if df.isnull().values.any():
        print("\nHandling missing values...")
        df.fillna(df.mean(), inplace=True)

    # Encoding the target column
    label_encoder = LabelEncoder()
    df['Attack_Name_Encoded'] = label_encoder.fit_transform(df['Attack Name'])

    # Splitting features and target
    X = df.drop(columns=['Attack Name', 'Attack_Name_Encoded'])
    y = df['Attack_Name_Encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Normalizing features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Hoeffding Tree Classifier
    model = HoeffdingTreeClassifier()

    print("\nTraining the model...")
    # Training the model one sample at a time
    for i in range(len(X_train)):
        model.learn_one(dict(enumerate(X_train[i])), y_train[i])

    # Evaluate the model
    y_pred = []
    for i in range(len(X_test)):
        pred = model.predict_one(dict(enumerate(X_test[i])))
        y_pred.append(pred)

    print("\nModel evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save the trained model
    save_model(model, scaler, label_encoder)
    
    return model, scaler, label_encoder

def save_model(model, scaler, label_encoder):
    """Save the trained model and its components"""
    # Create models directory if it doesn't exist
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and components
    model_path = f'trained_models/hoeffding_tree_model_{timestamp}.pkl'
    scaler_path = f'trained_models/hoeffding_tree_scaler_{timestamp}.joblib'
    encoder_path = f'trained_models/hoeffding_tree_encoder_{timestamp}.joblib'
    
    # Save Hoeffding Tree with pickle (as it's not compatible with joblib)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler and encoder with joblib
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\nModel saved successfully!")
    print(f"Model path: {model_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Encoder path: {encoder_path}")

if __name__ == "__main__":
    model, scaler, label_encoder = train_hoeffding_tree_model()