from sklearn.neural_network import MLPRegressor

def create_model():
    # MLPRegressor is a Neural Network akin to the Dense layers in Keras
    # Hidden layers: (64, 32, 16) matching the previous architecture
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        verbose=True
    )
    return model
