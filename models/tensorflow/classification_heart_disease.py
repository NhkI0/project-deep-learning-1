import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
import os


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    """Load and prepare data for Heart Disease classification."""
    df = pd.read_csv(csv_path)

    y = df['Heart_Disease_Yes'].values
    X = df.drop('Heart_Disease_Yes', axis=1).values
    feature_names = df.drop('Heart_Disease_Yes', axis=1).columns.tolist()

    return X, y, feature_names


def create_classification_model(input_dim):
    """
    Create classification model with architecture matching PyTorch configuration.

    Architecture: 256 -> 128 -> 64 units
    Aggressive regularization:
    - Dropout: 0.6 (to counter SMOTE overfitting)
    - L2 penalty: 1e-4
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),

        # Layer 1: 256 units
        keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        ),
        keras.layers.Dropout(0.6),

        # Layer 2: 128 units
        keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        ),
        keras.layers.Dropout(0.6),

        # Layer 3: 64 units
        keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        ),
        keras.layers.Dropout(0.6),

        # Output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Recall(name='recall')]
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return history


def find_optimal_threshold(y_true, y_pred_proba, target_recall=0.85):
    """
    Find optimal threshold to achieve target recall.
    Medical screening requires high recall (sensitivity) to catch most cases.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_diff = float('inf')

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)

        results.append({
            'threshold': threshold,
            'recall': recall,
            'precision': precision
        })

        # Find threshold closest to target recall
        diff = abs(recall - target_recall)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

    return best_threshold, results


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model with specified threshold."""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > threshold).astype(int).flatten()

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

    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp} (acceptable trade-off for better screening)")
    print(f"False Negatives: {fn} (minimize this for medical screening)")
    print(f"True Positives: {tp}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

    return y_pred, y_pred_proba


def main():
    """
    Main training pipeline following PyTorch configuration:
    - StandardScaler normalization
    - SMOTE with sampling_strategy=0.8 (creates ~44% positive cases)
    - Architecture: 256 -> 128 -> 64 with 0.6 dropout and L2=1e-4
    - Threshold calibration for 85% recall target
    """
    print("="*60)
    print("HEART DISEASE CLASSIFICATION - TensorFlow/Keras")
    print("="*60)

    print("\nLoading data...")
    X, y, feature_names = load_and_prepare_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Original class distribution: {np.bincount(y)} ({y.mean()*100:.1f}% positive)")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Normalize with StandardScaler
    print("\nNormalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE ONLY on training set with sampling_strategy=0.8
    print("\nApplying SMOTE on training set (sampling_strategy=0.8)...")
    print(f"Before SMOTE - Training: {X_train_scaled.shape}, Positive: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"After SMOTE - Training: {X_train_resampled.shape}, "
          f"Positive: {y_train_resampled.sum()} ({y_train_resampled.mean()*100:.1f}%)")
    print(f"⚠️  Validation and test sets remain untouched (no SMOTE)")

    print(f"\nFinal dataset sizes:")
    print(f"  Training: {X_train_resampled.shape}")
    print(f"  Validation: {X_val_scaled.shape}")
    print(f"  Test: {X_test_scaled.shape}")

    # Create model
    print("\nCreating model...")
    print("Architecture: 256 -> 128 -> 64 (Dropout=0.6, L2=1e-4)")
    model = create_classification_model(input_dim=X_train_resampled.shape[1])
    model.summary()

    # Train model
    print("\nTraining model...")
    history = train_model(
        model,
        X_train_resampled, y_train_resampled,
        X_val_scaled, y_val,
        epochs=100,
        batch_size=32
    )

    # Find optimal threshold on validation set
    print("\n" + "="*60)
    print("THRESHOLD CALIBRATION (on validation set)")
    print("="*60)

    y_val_pred_proba = model.predict(X_val_scaled)
    optimal_threshold, threshold_results = find_optimal_threshold(
        y_val, y_val_pred_proba, target_recall=0.85
    )

    print(f"\nOptimal threshold for 85% recall target: {optimal_threshold:.3f}")
    print("(Default threshold of 0.5 favors precision over recall)")

    # Evaluate with default threshold (0.5)
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

    model_path = 'saved_models/heart_disease_classification.keras'
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    import joblib
    scaler_path = 'saved_models/heart_disease_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    threshold_path = 'saved_models/optimal_threshold.txt'
    with open(threshold_path, 'w') as f:
        f.write(f"{optimal_threshold:.4f}")
    print(f"✓ Optimal threshold saved to {threshold_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nConfiguration (matching PyTorch):")
    print(f"  - SMOTE sampling_strategy: 0.8")
    print(f"  - Architecture: 256 -> 128 -> 64")
    print(f"  - Dropout: 0.6")
    print(f"  - L2 regularization: 1e-4")
    print(f"  - Learning rate: 0.001")
    print(f"  - Optimal threshold: {optimal_threshold:.3f} (for 85% recall)")

    return model, history, optimal_threshold


if __name__ == "__main__":
    main()
