import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class CancerPredictionModel:
    def __init__(self):
        self.data = load_breast_cancer()
        self.model = None
        self.feature_names = self.data.feature_names
        self.target_names = self.data.target_names
        self._train_model()

    def _train_model(self):
        """Loads dataset and trains the Machine Learning model."""
        df = pd.DataFrame(self.data.data, columns=self.feature_names)
        df["target"] = self.data.target
        
        X = df.drop("target", axis=1)
        y = df["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def get_feature_stats(self):
        """Returns feature min, max, and mean for setting up UI controls."""
        stats = {}
        for idx, feature in enumerate(self.feature_names):
            stats[feature] = {
                "min": float(np.min(self.data.data[:, idx])),
                "max": float(np.max(self.data.data[:, idx])),
                "mean": float(np.mean(self.data.data[:, idx]))
            }
        return stats

    def predict(self, input_dict):
        """Accepts feature dict, runs prediction, returns outcome and confidence."""
        input_df = pd.DataFrame([input_dict])
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        
        # 0 = Malignant, 1 = Benign in Scikit-Learn's dataset
        is_benign = (prediction == 1)
        confidence = probabilities[1] if is_benign else probabilities[0]
        
        return {
            "is_benign": is_benign,
            "label": "Benign (Non-Cancerous)" if is_benign else "Malignant (Potentially Cancerous)",
            "confidence": round(confidence * 100, 2)
        }