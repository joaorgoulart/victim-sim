import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DecisionTreeModel:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.scaler = StandardScaler()
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
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

if __name__ == "__main__":
    # Carregar e preparar os dados
    X, y = load_and_prepare_data("datasets/data_4000v.txt")
    
    # Criar e treinar o modelo
    dt_model = DecisionTreeModel()
    trained_model = train_and_evaluate_model(dt_model, X, y)
    
    # Salvar o modelo treinado (opcional)
    # joblib.dump(trained_model, 'trained_dt_model.joblib')