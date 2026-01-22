import joblib
import numpy as np
import sys

def predict_pressure(balls_remaining, runs_needed, wickets_left, target_score):
    # Calculate derived feature RRR
    if balls_remaining > 0:
        rrr = (runs_needed / balls_remaining) * 6
    else:
        rrr = 999 
        
    features = np.array([[balls_remaining, runs_needed, wickets_left, rrr, target_score]])
    
    # Load scaler and model
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('pressure_model.pkl')
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return
        
    # Scale input
    features_scaled = scaler.transform(features)
    
    # Predict
    pressure = model.predict(features_scaled)[0]
    
    # Clamp to 0-10
    pressure = max(0, min(10, pressure))
    
    print(f"\n--- Match State ---")
    print(f"Target: {target_score}")
    print(f"Need {runs_needed} runs in {balls_remaining} balls with {wickets_left} wickets left")
    print(f"Required Run Rate: {rrr:.2f}")
    print(f"-------------------")
    print(f"PREDICTED PRESSURE: {pressure:.2f}/10")
    print(f"-------------------")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        # Example: python predict.py 30 50 6 180
        # balls_rem, runs_need, wkt_left, target
        balls_remaining = int(sys.argv[1])
        runs_needed = int(sys.argv[2])
        wickets_left = int(sys.argv[3])
        target_score = int(sys.argv[4])
        print(f"Running prediction for: {sys.argv[1:]}")
        predict_pressure(balls_remaining, runs_needed, wickets_left, target_score)
    else:
        print("Usage: python predict.py <balls_remaining> <runs_needed> <wickets_left> <target_score>")
        print("Running demo scenarios...")
        
        # Scenario 1: High Pressure
        predict_pressure(16, 20, 3, 200)
        
        # Scenario 2: Low Pressure
        predict_pressure(18, 40, 8, 180)
        
        # Scenario 3: Medium Pressure
        predict_pressure(30, 25, 6, 170)
