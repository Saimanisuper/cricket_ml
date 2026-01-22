import pandas as pd
import numpy as np
import random

def calculate_pressure(run_rate_req, wickets_left, balls_remaining):
    """
    Heuristic to calculate pressure (0-10 scale).
    High pressure: High RRR, Low wickets, Low balls remaining.
    """
    if balls_remaining == 0:
        return 10.0 # Maximum pressure if balls finished and target not met (handled by logic usually)
        
    # Pressure factors
    rrr_factor = min(run_rate_req / 36.0, 1.0) * 10  # Normalized RRR contribution
    
    # Wickets factor: fewer wickets = more pressure
    wickets_factor = (10 - wickets_left) / 10.0 * 5 
    
    # Balls factor: fewer balls = more pressure (exponentially)
    balls_factor = (120 - balls_remaining) / 120.0 * 3
    
    # Combined pressure
    pressure = rrr_factor + wickets_factor + balls_factor
    
    # Add some randomness/noise
    noise = np.random.normal(0, 0.5)
    pressure += noise
    
    return min(max(pressure, 0.0), 10.0)

def generate_match_data(num_samples=10000):
    data = []
    
    for _ in range(num_samples):
        # Initial match parameters
        target_score = random.randint(120, 240) # T20 scores
        
        # Simulate a random state in the 2nd innings
        balls_bowled = random.randint(1, 119)
        balls_remaining = 120 - balls_bowled
        
        # Runs scored so far (heuristically related to balls bowled but with variance)
        max_possible_runs = min(target_score + 6, balls_bowled * 6) # theoretical max
        avg_run_rate = target_score / 120.0
        
        # Randomly decide current performance (good, bad, average)
        performance_factor = random.uniform(0.5, 1.5)
        current_runs = int(balls_bowled * avg_run_rate * performance_factor)
        current_runs = min(current_runs, target_score - 1) # Ensure match isn't won yet
        
        # Wickets lost
        # More balls bowled -> higher probability of more wickets lost
        wickets_prob = balls_bowled / 120.0
        wickets_lost = np.random.binomial(9, wickets_prob) # max 9 wickets down (if 10, all out)
        wickets_left = 10 - wickets_lost
        
        runs_needed = target_score - current_runs
        rrr = (runs_needed / balls_remaining) * 6
        
        pressure = calculate_pressure(rrr, wickets_left, balls_remaining)
        
        data.append({
            'balls_remaining': balls_remaining,
            'runs_needed': runs_needed,
            'wickets_left': wickets_left,
            'rrr': rrr,
            'target_score': target_score,
            'pressure': pressure
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_match_data(50000)
    output_file = "cricket_pressure_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} samples and saved to {output_file}")
    print(df.head())
