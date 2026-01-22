# Cricket Match Pressure Predictor

A machine learning project to predict the "pressure" index in a cricket match chase scenario.

## Overview

This project simulates 2nd innings cricket match scenarios (T20 format) and calculates a heuristic "pressure" score based on:
-   Required Run Rate (RRR)
-   Wickets Left
-   Balls Remaining

It uses a Multi-Layer Perceptron (MLPRegressor) to learn this relationship and predict pressure for new scenarios.

## Project Structure

-   `data_generator.py`: Generates synthetic match data and saves it to `cricket_pressure_data.csv`.
-   `model.py`: Defines the Neural Network architecture (sklearn MLPRegressor).
-   `train.py`: Trains the model on the generated data and saves artifacts (`pressure_model.pkl`, `scaler.pkl`).
-   `predict.py`: CLI tool to predict pressure for a given match state.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Saimanisuper/cricket_ml.git
    cd cricket_ml
    ```

2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn joblib
    ```

## Usage

### 1. (Optional) Generate Data
If you want to regenerate the dataset:
```bash
python data_generator.py
```

### 2. Train the Model
Train the neural network:
```bash
python train.py
```
This will create `pressure_model.pkl` and `scaler.pkl`.

### 3. Make Predictions
Run the predictor with specific match parameters:
```bash
python predict.py <balls_remaining> <runs_needed> <wickets_left> <target_score>
```

**Example:**
Need 50 runs in 30 balls with 6 wickets left (Target 180):
```bash
python predict.py 30 50 6 180
```
