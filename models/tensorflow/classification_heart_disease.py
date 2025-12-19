import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, recall_score, precision_score, roc_curve,
                             precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrixCallback(keras.callbacks.Callback):
    """
    Callback to generate confusion matrix during training.

    Parameters:
    -----------
    X_val : array
        Validation features
    y_val : array
        Validation labels
    threshold : float
        Classification threshold (default: 0.5)
    save_freq : int
        Save confusion matrix figure every N epochs (default: 10, 0 = never save)
    figures_dir : str
        Directory to save confusion matrix figures
    """

    def __init__(self, X_val, y_val, threshold=0.5, save_freq=10, figures_dir='figures'):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.threshold = threshold
        self.save_freq = save_freq
        self.figures_dir = figures_dir
        os.makedirs(f'{figures_dir}/training_cm', exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_proba > self.threshold).astype(int).flatten()

        # Compute confusion matrix
        cm = confusion_matrix(self.y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        recall = recall_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred, zero_division=0)
        accuracy = accuracy_score(self.y_val, y_pred)

        # Print metrics
        print(f"\n[Epoch {epoch + 1}] Confusion Matrix (threshold={self.threshold}):")
        print(f"  TN: {tn:4d} | FP: {fp:4d}")
        print(f"  FN: {fn:4d} | TP: {tp:4d}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")

        # Save confusion matrix figure at specified frequency
        if self.save_freq > 0 and (epoch + 1) % self.save_freq == 0:
            self._save_confusion_matrix(cm, epoch + 1, recall, precision, accuracy)

    def _save_confusion_matrix(self, cm, epoch, recall, precision, accuracy):
        """Save confusion matrix figure."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=11)
        plt.ylabel('Actual', fontsize=11)
        plt.title(f'Confusion Matrix - Epoch {epoch}\n'
                  f'Precision: {precision:.3f} | Recall: {recall:.3f} | Accuracy: {accuracy:.3f}',
                  fontsize=12)
        plt.tight_layout()
        save_path = f'{self.figures_dir}/training_cm/confusion_matrix_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Saved confusion matrix to {save_path}")


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


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                use_confusion_matrix=True, cm_threshold=0.5, cm_save_freq=10,
                figures_dir='figures'):
    """
    Train the model with optional confusion matrix monitoring.

    Parameters:
    -----------
    use_confusion_matrix : bool
        Whether to compute confusion matrix during training (default: True)
    cm_threshold : float
        Threshold for confusion matrix classification (default: 0.5)
    cm_save_freq : int
        Save confusion matrix figure every N epochs (default: 10, 0 = only print)
    figures_dir : str
        Directory to save confusion matrix figures
    """
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

    callbacks = [early_stopping, reduce_lr]

    # Add confusion matrix callback if requested
    if use_confusion_matrix:
        cm_callback = ConfusionMatrixCallback(
            X_val, y_val,
            threshold=cm_threshold,
            save_freq=cm_save_freq,
            figures_dir=figures_dir
        )
        callbacks.append(cm_callback)
        print(f"\n✓ Confusion matrix monitoring enabled (threshold={cm_threshold}, save every {cm_save_freq} epochs)")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
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


def plot_correlation_matrix(df, target_name='Heart_Disease_Yes', figures_dir='figures'):
    """Plot and save correlation matrix of features and target."""
    os.makedirs(figures_dir, exist_ok=True)

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Plot full correlation matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved full correlation matrix to {figures_dir}/correlation_matrix_full.png")

    # Plot correlation with target only
    target_corr = corr_matrix[target_name].sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    top_features = target_corr.head(20)
    colors = ['red' if x < 0 else 'green' for x in top_features.values]
    plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features.index, fontsize=9)
    plt.xlabel('Correlation Coefficient', fontsize=11)
    plt.title(f'Top 20 Features Correlated with {target_name}', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/correlation_with_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved target correlations to {figures_dir}/correlation_with_target.png")


def plot_training_history(history, figures_dir='figures'):
    """Plot and save training history."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Model Loss Over Epochs', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Model Accuracy Over Epochs', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('AUC', fontsize=11)
    axes[1, 0].set_title('Model AUC Over Epochs', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Recall', fontsize=11)
    axes[1, 1].set_title('Model Recall Over Epochs', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {figures_dir}/training_history.png")


def plot_confusion_matrices(y_test, y_pred_default, y_pred_optimal, threshold_default=0.5,
                            threshold_optimal=0.5, figures_dir='figures'):
    """Plot confusion matrices for both thresholds."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Default threshold confusion matrix
    cm_default = confusion_matrix(y_test, y_pred_default)
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].set_title(f'Confusion Matrix (Threshold={threshold_default:.2f})', fontsize=12)

    # Optimal threshold confusion matrix
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].set_title(f'Confusion Matrix (Threshold={threshold_optimal:.2f})', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to {figures_dir}/confusion_matrices.png")


def plot_roc_curve(y_test, y_pred_proba, figures_dir='figures'):
    """Plot ROC curve."""
    os.makedirs(figures_dir, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {figures_dir}/roc_curve.png")


def plot_precision_recall_curve(y_test, y_pred_proba, figures_dir='figures'):
    """Plot Precision-Recall curve."""
    os.makedirs(figures_dir, exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall Curve', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision-Recall curve to {figures_dir}/precision_recall_curve.png")


def plot_threshold_analysis(threshold_results, optimal_threshold, figures_dir='figures'):
    """Plot threshold analysis showing precision-recall tradeoff."""
    os.makedirs(figures_dir, exist_ok=True)

    thresholds = [r['threshold'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, recalls, label='Recall', linewidth=2, color='blue')
    plt.plot(thresholds, precisions, label='Precision', linewidth=2, color='green')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.axhline(y=0.85, color='orange', linestyle='--', linewidth=1,
                label='Target Recall (0.85)', alpha=0.7)
    plt.xlabel('Threshold', fontsize=11)
    plt.ylabel('Score', fontsize=11)
    plt.title('Precision-Recall vs Threshold', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold analysis to {figures_dir}/threshold_analysis.png")


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

    # Create figures directory
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    # Load full dataframe for correlation analysis
    print("\nGenerating correlation matrix...")
    df = pd.read_csv('../../data/CVD_cleaned_dummies.csv')
    plot_correlation_matrix(df, target_name='Heart_Disease_Yes', figures_dir=figures_dir)

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
        batch_size=32,
        use_confusion_matrix=True,
        cm_threshold=0.5,
        cm_save_freq=10,
        figures_dir=figures_dir
    )

    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, figures_dir)

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
    y_pred_default, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, threshold=0.5)

    # Evaluate with optimal threshold
    print("\n" + "="*60)
    print(f"EVALUATION WITH OPTIMAL THRESHOLD ({optimal_threshold:.3f})")
    print("="*60)
    print("✓ Calibrated for medical screening (85% recall target)")
    y_pred_optimal, _ = evaluate_model(model, X_test_scaled, y_test, threshold=optimal_threshold)

    # Generate all visualization plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60)
    plot_confusion_matrices(y_test, y_pred_default, y_pred_optimal,
                           threshold_default=0.5, threshold_optimal=optimal_threshold,
                           figures_dir=figures_dir)
    plot_roc_curve(y_test, y_pred_proba, figures_dir)
    plot_precision_recall_curve(y_test, y_pred_proba, figures_dir)
    plot_threshold_analysis(threshold_results, optimal_threshold, figures_dir)

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

    print("\n" + "="*60)
    print("ALL FIGURES SAVED TO:", figures_dir)
    print("="*60)
    print("Generated figures:")
    print(f"  - {figures_dir}/correlation_matrix_full.png")
    print(f"  - {figures_dir}/correlation_with_target.png")
    print(f"  - {figures_dir}/training_history.png")
    print(f"  - {figures_dir}/confusion_matrices.png")
    print(f"  - {figures_dir}/roc_curve.png")
    print(f"  - {figures_dir}/precision_recall_curve.png")
    print(f"  - {figures_dir}/threshold_analysis.png")
    print(f"  - {figures_dir}/training_cm/ (confusion matrices during training)")

    return model, history, optimal_threshold


if __name__ == "__main__":
    main()
