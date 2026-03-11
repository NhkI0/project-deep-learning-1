"""
Generate Predictions Dataset (X + Y') for Heart Disease Classification

Usage:
    python generate_predictions.py

Output:
    - predictions/dataset_with_predictions.csv (input features + predicted output)
"""

import os
import pandas as pd
import joblib
from tensorflow import keras


def main():
    os.makedirs('predictions', exist_ok=True)

    df = pd.read_csv('data/CVD_cleaned_dummies.csv')
    df = df.astype(float)
    X_df = df.drop('Heart_Disease_Yes', axis=1)

    base_path = 'models/tensorflow/saved_models'
    model = keras.models.load_model(f'{base_path}/heart_disease_classification.keras')
    scaler = joblib.load(f'{base_path}/heart_disease_scaler.pkl')

    X_scaled = scaler.transform(X_df.values)
    y_pred_proba = model.predict(X_scaled, verbose=0).flatten()

    result = X_df.copy()
    result['Heart_Disease_Predicted'] = y_pred_proba

    result.to_csv('predictions/dataset_with_predictions.csv', index=False)
    print(f"fini {len(result)}")


if __name__ == '__main__':
    main()
