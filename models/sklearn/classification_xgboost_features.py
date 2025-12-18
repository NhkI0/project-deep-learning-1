"""
Scikit-learn Heart Disease Classification with XGBoost-Style Feature Engineering

This model replicates the feature engineering approach from the XGBoost notebook:
- Extensive feature engineering (BMI categories, lifestyle scores, interactions)
- MinMaxScaler (0-1 range) instead of StandardScaler
- SMOTE + TomekLinks hybrid resampling
- MLPClassifier architecture (256->128->64)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import joblib
import os


def engineer_features(data):
    """
    Feature engineering following XGBoost notebook approach EXACTLY.
    Creates derived features and interaction terms.
    """
    data = data.copy()

    # 1. BMI Category (categorical labels, will be mapped to ordinal in preprocessing)
    data['BMI_Category'] = pd.cut(
        data['BMI'],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    )

    # 2. Checkup Frequency (ordinal mapping)
    checkup_mapping = {
        'Within the past year': 4,
        'Within the past 2 years': 2,
        'Within the past 5 years': 1,
        '5 or more years ago': 0.2,
        'Never': 0
    }
    data['Checkup_Frequency'] = data['Checkup'].replace(checkup_mapping)

    # 3. Lifestyle Score (composite: exercise + diet - smoking - alcohol)
    exercise_mapping = {'Yes': 1, 'No': 0}
    smoking_mapping = {'Yes': -1, 'No': 0}

    data['Lifestyle_Score'] = (
        data['Exercise'].replace(exercise_mapping)
        - data['Smoking_History'].replace(smoking_mapping)
        + data['Fruit_Consumption']/10
        + data['Green_Vegetables_Consumption']/10
        - data['Alcohol_Consumption']/10
    )

    # 4. Healthy Diet Score
    data['Healthy_Diet_Score'] = (
        data['Fruit_Consumption']/10
        + data['Green_Vegetables_Consumption']/10
        - data['FriedPotato_Consumption']/10
    )

    # 5. Interaction Terms
    data['Smoking_Alcohol'] = (
        data['Smoking_History'].replace(smoking_mapping) * data['Alcohol_Consumption']
    )

    data['Checkup_Exercise'] = (
        data['Checkup_Frequency'] * data['Exercise'].replace(exercise_mapping)
    )

    # 6. Height to Weight Ratio
    data['Height_to_Weight'] = data['Height_(cm)'] / data['Weight_(kg)']

    # 7. Fruit and Vegetables Interaction
    data['Fruit_Vegetables'] = (
        data['Fruit_Consumption'] * data['Green_Vegetables_Consumption']
    )

    # 8. Healthy Diet × Lifestyle Interaction
    data['HealthyDiet_Lifestyle'] = (
        data['Healthy_Diet_Score'] * data['Lifestyle_Score']
    )

    # 9. Alcohol × Fried Potato Interaction
    data['Alcohol_FriedPotato'] = (
        data['Alcohol_Consumption'] * data['FriedPotato_Consumption']
    )

    return data


def preprocess_data(data):
    """
    Preprocess data following XGBoost notebook approach EXACTLY.
    Includes ordinal encoding and binary mapping.
    """
    data = data.copy()

    # Diabetes mapping (combine categories)
    diabetes_mapping = {
        'No': 0,
        'No, pre-diabetes or borderline diabetes': 0,
        'Yes, but female told only during pregnancy': 1,
        'Yes': 1
    }
    data['Diabetes'] = data['Diabetes'].map(diabetes_mapping)

    # One-hot encoding for Sex
    data = pd.get_dummies(data, columns=['Sex'])

    # Binary columns
    binary_columns = [
        'Heart_Disease', 'Skin_Cancer', 'Other_Cancer',
        'Depression', 'Arthritis', 'Smoking_History', 'Exercise'
    ]
    for column in binary_columns:
        data[column] = data[column].map({'Yes': 1, 'No': 0})

    # Ordinal encoding for General_Health
    general_health_mapping = {
        'Poor': 0,
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Excellent': 4
    }
    data['General_Health'] = data['General_Health'].map(general_health_mapping)

    # Ordinal encoding for BMI_Category
    bmi_mapping = {
        'Underweight': 0,
        'Normal weight': 1,
        'Overweight': 2,
        'Obesity': 3
    }
    data['BMI_Category'] = data['BMI_Category'].map(bmi_mapping).astype(int)

    # Ordinal encoding for Age_Category
    age_category_mapping = {
        '18-24': 0,
        '25-29': 1,
        '30-34': 2,
        '35-39': 3,
        '40-44': 4,
        '45-49': 5,
        '50-54': 6,
        '55-59': 7,
        '60-64': 8,
        '65-69': 9,
        '70-74': 10,
        '75-79': 11,
        '80+': 12
    }
    data['Age_Category'] = data['Age_Category'].map(age_category_mapping)

    # Drop Checkup column
    data = data.drop(["Checkup"], axis=1)

    return data


def load_and_prepare_data(csv_path='../../data/CVD_cleaned.csv'):
    """Load data and apply feature engineering + preprocessing."""
    print("Loading data...")
    data = pd.read_csv(csv_path)

    print("Applying feature engineering...")
    data = engineer_features(data)

    print("Preprocessing features...")
    data = preprocess_data(data)

    # Remove duplicates
    data = data.drop_duplicates()

    # Separate features and target
    y = data['Heart_Disease'].values
    X = data.drop('Heart_Disease', axis=1).values
    feature_names = data.drop('Heart_Disease', axis=1).columns.tolist()

    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features include engineered: BMI_Category, Lifestyle_Score, interactions, etc.")

    return X, y, feature_names


def create_mlp_classifier(random_state=42):
    """
    Create MLPClassifier with architecture matching TensorFlow version.
    Architecture: 256 -> 128 -> 64 with aggressive regularization.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        alpha=1e-4,  # L2 regularization
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=random_state,
        verbose=True
    )
    return model


def train_model(model, X_train, y_train):
    """Train the model."""
    print("\nTraining model...")
    model.fit(X_train, y_train)
    return model


def find_optimal_threshold(y_true, y_pred_proba, target_recall=0.85):
    """Find optimal threshold to achieve target recall."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_diff = float('inf')

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        recall = recall_score(y_true, y_pred)

        diff = abs(recall - target_recall)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

    return best_threshold


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model with specified threshold."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    print("\n" + "="*60)
    print(f"MODEL EVALUATION - Threshold: {threshold:.2f}")
    print("="*60)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f} ({'✓' if recall >= 0.85 else '✗'} target: 0.85)")
    print(f"Precision: {precision:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

    return y_pred, y_pred_proba


def main():
    """
    Main training pipeline using XGBoost notebook approach:
    - Extensive feature engineering (10+ derived features)
    - MinMaxScaler (0-1 normalization)
    - SMOTE + TomekLinks hybrid resampling
    - MLPClassifier: 256->128->64
    - Threshold calibration for 85% recall
    """
    print("="*60)
    print("HEART DISEASE CLASSIFICATION - XGBoost Features (sklearn)")
    print("="*60)

    # Load data with feature engineering
    X, y, feature_names = load_and_prepare_data()

    print(f"\nOriginal class distribution: {np.bincount(y)} ({y.mean()*100:.1f}% positive)")

    # Split data - Following XGBoost notebook: 0.2 test set, then split remaining for train/val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    # Normalize with MinMaxScaler (0-1 range, XGBoost approach)
    print("\nNormalizing features with MinMaxScaler (0-1 range)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE (oversample minority) + TomekLinks (undersample majority)
    print("\nApplying hybrid resampling: SMOTE + TomekLinks...")
    print(f"Before resampling - Training: {X_train_scaled.shape}, "
          f"Positive: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    # SMOTE: oversample minority class
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE - Training: {X_train_smote.shape}, "
          f"Positive: {y_train_smote.sum()} ({y_train_smote.mean()*100:.1f}%)")

    # TomekLinks: clean borders by removing majority class samples
    tomek = TomekLinks(sampling_strategy='majority')
    X_train_resampled, y_train_resampled = tomek.fit_resample(X_train_smote, y_train_smote)
    print(f"After TomekLinks - Training: {X_train_resampled.shape}, "
          f"Positive: {y_train_resampled.sum()} ({y_train_resampled.mean()*100:.1f}%)")

    print(f"\n⚠️  Validation and test sets remain untouched")
    print(f"\nFinal dataset sizes:")
    print(f"  Training: {X_train_resampled.shape}")
    print(f"  Validation: {X_val_scaled.shape}")
    print(f"  Test: {X_test_scaled.shape}")

    # Create model
    print("\nCreating model...")
    print("Architecture: 256 -> 128 -> 64 (L2=1e-4)")
    model = create_mlp_classifier()

    # Train model
    model = train_model(model, X_train_resampled, y_train_resampled)

    # Find optimal threshold
    print("\n" + "="*60)
    print("THRESHOLD CALIBRATION (on validation set)")
    print("="*60)

    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_val_pred_proba, target_recall=0.85)

    print(f"\nOptimal threshold for 85% recall target: {optimal_threshold:.3f}")

    # Evaluate with default threshold
    print("\n" + "="*60)
    print("EVALUATION WITH DEFAULT THRESHOLD (0.5)")
    print("="*60)
    evaluate_model(model, X_test_scaled, y_test, threshold=0.5)

    # Evaluate with optimal threshold
    print("\n" + "="*60)
    print(f"EVALUATION WITH OPTIMAL THRESHOLD ({optimal_threshold:.3f})")
    print("="*60)
    print("✓ Calibrated for medical screening (85% recall target)")
    evaluate_model(model, X_test_scaled, y_test, threshold=optimal_threshold)

    # Save model and artifacts
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)

    os.makedirs('saved_models', exist_ok=True)

    model_path = 'saved_models/heart_disease_xgboost_features_sklearn.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")

    scaler_path = 'saved_models/heart_disease_xgboost_scaler_sklearn.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    threshold_path = 'saved_models/optimal_threshold_xgboost_sklearn.txt'
    with open(threshold_path, 'w') as f:
        f.write(f"{optimal_threshold:.4f}")
    print(f"✓ Optimal threshold saved to {threshold_path}")

    # Save feature names for reference
    features_path = 'saved_models/xgboost_feature_names_sklearn.txt'
    with open(features_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    print(f"✓ Feature names saved to {features_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nConfiguration (XGBoost notebook approach):")
    print(f"  - Feature engineering: 10+ derived features")
    print(f"  - Scaler: MinMaxScaler (0-1 range)")
    print(f"  - Resampling: SMOTE + TomekLinks")
    print(f"  - Architecture: 256 -> 128 -> 64")
    print(f"  - L2 regularization: 1e-4")
    print(f"  - Learning rate: 0.001")
    print(f"  - Optimal threshold: {optimal_threshold:.3f} (for 85% recall)")

    print(f"\nEngineered Features:")
    print(f"  - BMI_Category, Checkup_Frequency")
    print(f"  - Lifestyle_Score, Healthy_Diet_Score")
    print(f"  - Smoking_Alcohol, Checkup_Exercise")
    print(f"  - Height_to_Weight, Fruit_Vegetables")
    print(f"  - HealthyDiet_Lifestyle, Alcohol_FriedPotato")

    return model, scaler, optimal_threshold


if __name__ == "__main__":
    main()
