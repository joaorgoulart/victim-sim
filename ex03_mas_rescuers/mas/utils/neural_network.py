import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetworkModel:
    def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )

    def preprocess_data(self, X):
        return self.scaler.transform(X)

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        X_scaled = self.preprocess_data(X)
        return self.model.score(X_scaled, y)

def load_and_prepare_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, [3, 4, 5]]  # qPA, pulso, frequência respiratória
    y = data[:, 6]  # gravidade
    return X, y

def train_and_evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
    
    train_score = model.evaluate(X_train, y_train)
    test_score = model.evaluate(X_test, y_test)
    
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    return model