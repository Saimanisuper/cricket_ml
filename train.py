import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
from model import create_model

def train():
    # Load data
    df = pd.read_csv("cricket_pressure_data.csv")
    
    # Feature selection
    feature_cols = ['balls_remaining', 'runs_needed', 'wickets_left', 'rrr', 'target_score']
    X = df[feature_cols].values
    y = df['pressure'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for inference
    joblib.dump(scaler, 'scaler.pkl')
    
    # Create and train model
    print("Training MLP Regressor...")
    model = create_model()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {mae}")
    
    # Save model
    joblib.dump(model, 'pressure_model.pkl')
    print("Model saved to pressure_model.pkl")

if __name__ == "__main__":
    train()
