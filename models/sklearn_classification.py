import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier


class SklearnClassification:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_fitted = False
        self.model = RandomForestClassifier(random_state=self.random_state)
        self.is_fitted = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting. Call train() first.")
        return self.model.predict(X)


    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting. Call train() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting. Call train() first.")
        y_pred = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Add ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2:
            y_proba = self.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)

        return metrics
