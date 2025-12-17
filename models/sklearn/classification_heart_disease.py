import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    df = pd.read_csv(csv_path)

    y = df['Heart_Disease_Yes'].values

    X = df.drop('Heart_Disease_Yes', axis=1).values
    feature_names = df.drop('Heart_Disease_Yes', axis=1).columns.tolist()

    return X, y, feature_names


def create_mlp_classifier(input_dim, random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=random_state,
        verbose=True
    )
    return model


def train_model(model, X_train, y_train):
    print("\nTraining model...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n" + "="*50)
    print(f"MODEL EVALUATION - {model_name}")
    print("="*50)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

    return y_pred, y_pred_proba


def main():
    print("Loading data...")
    X, y, feature_names = load_and_prepare_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    print("\nCreating MLPClassifier (Neural Network)...")
    model = create_mlp_classifier(input_dim=X_train.shape[1])
    model = train_model(model, X_train, y_train)

    print("\nEvaluating model on test set...")
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, "MLPClassifier")

    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/heart_disease_classification_sklearn.pkl'
    scaler_path = 'saved_models/heart_disease_scaler_sklearn.pkl'

    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return model, scaler


if __name__ == "__main__":
    main()
