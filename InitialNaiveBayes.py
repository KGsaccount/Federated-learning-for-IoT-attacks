# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def train_naive_bayes_model():
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

    # Split features and target
    X = df.drop(columns=['Attack Name', 'Attack_Name_Encoded'])
    y = df['Attack_Name_Encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Naive Bayes Classifier
    model = GaussianNB()

    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-validation accuracy scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

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
    model_path = f'trained_models/naive_bayes_model_{timestamp}.joblib'
    scaler_path = f'trained_models/naive_bayes_scaler_{timestamp}.joblib'
    encoder_path = f'trained_models/naive_bayes_encoder_{timestamp}.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\nModel saved successfully!")
    print(f"Model path: {model_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Encoder path: {encoder_path}")

if __name__ == "__main__":
    model, scaler, label_encoder = train_naive_bayes_model()