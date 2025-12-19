import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    """Load and prepare data for Height and Weight regression."""
    df = pd.read_csv(csv_path)

    # Target variables
    y = df[['Height_(cm)', 'Weight_(kg)']].values

    # Features (everything except the targets)
    X = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).values
    feature_names = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).columns.tolist()
    target_names = ['Height_(cm)', 'Weight_(kg)']

    return X, y, feature_names, target_names


def create_multioutput_regression_model(input_dim, output_dim=2):
    """Create an improved multi-output regression model with BatchNorm and better architecture."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),

        # First block
        keras.layers.Dense(256, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),

        # Second block
        keras.layers.Dense(128, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),

        # Third block
        keras.layers.Dense(64, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.2),

        # Fourth block
        keras.layers.Dense(32, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.2),

        # Fifth block
        keras.layers.Dense(16, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        # Output layer
        keras.layers.Dense(output_dim, kernel_initializer='glorot_normal')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
    """Train the multi-output regression model with improved callbacks."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )

    # Cosine annealing for learning rate
    cosine_decay = keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * (np.cos(epoch * np.pi / epochs) + 1) / 2 + 1e-7
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


def evaluate_model(model, X_test, y_test, target_names, y_pred=None):
    """Evaluate the trained model."""
    if y_pred is None:
        y_pred = model.predict(X_test)

    print("\n" + "="*50)
    print("MODEL EVALUATION - Height & Weight Regression")
    print("="*50)

    for i, target_name in enumerate(target_names):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred[:, i]

        mse = mean_squared_error(y_true_i, y_pred_i)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_i, y_pred_i)
        r2 = r2_score(y_true_i, y_pred_i)

        print(f"\n{target_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")

    # Overall metrics
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_mae = mean_absolute_error(y_test, y_pred)

    print(f"\nOverall:")
    print(f"  MSE: {overall_mse:.4f}")
    print(f"  MAE: {overall_mae:.4f}")

    return y_pred


def plot_correlation_matrix(df, target_names, figures_dir='figures'):
    """Plot and save correlation matrix of features and targets."""
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

    # Plot correlation with targets only
    target_corr = corr_matrix[target_names].sort_values(by=target_names[0], ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for i, target in enumerate(target_names):
        top_features = target_corr[target].head(15)
        axes[i].barh(range(len(top_features)), top_features.values)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features.index, fontsize=9)
        axes[i].set_xlabel('Correlation Coefficient', fontsize=11)
        axes[i].set_title(f'Top 15 Features Correlated with {target}', fontsize=12)
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/correlation_with_targets.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved target correlations to {figures_dir}/correlation_with_targets.png")


def plot_training_history(history, figures_dir='figures'):
    """Plot and save training history."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (MSE)', fontsize=11)
    axes[0].set_title('Model Loss Over Epochs', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].set_title('Model MAE Over Epochs', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {figures_dir}/training_history.png")


def plot_predictions(y_test, y_pred, target_names, figures_dir='figures'):
    """Plot predictions vs actual values."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, target_name in enumerate(target_names):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred[:, i]

        # Scatter plot
        axes[i].scatter(y_true_i, y_pred_i, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_true_i.min(), y_pred_i.min())
        max_val = max(y_true_i.max(), y_pred_i.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Calculate R²
        r2 = r2_score(y_true_i, y_pred_i)

        axes[i].set_xlabel(f'Actual {target_name}', fontsize=11)
        axes[i].set_ylabel(f'Predicted {target_name}', fontsize=11)
        axes[i].set_title(f'{target_name} Predictions (R²={r2:.4f})', fontsize=12)
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions plot to {figures_dir}/predictions_vs_actual.png")


def plot_residuals(y_test, y_pred, target_names, figures_dir='figures'):
    """Plot residual analysis."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, target_name in enumerate(target_names):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred[:, i]
        residuals = y_true_i - y_pred_i

        # Residuals vs predictions
        axes[i, 0].scatter(y_pred_i, residuals, alpha=0.5, s=20)
        axes[i, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[i, 0].set_xlabel(f'Predicted {target_name}', fontsize=10)
        axes[i, 0].set_ylabel('Residuals', fontsize=10)
        axes[i, 0].set_title(f'Residuals vs Predicted - {target_name}', fontsize=11)
        axes[i, 0].grid(alpha=0.3)

        # Residuals distribution
        axes[i, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[i, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[i, 1].set_xlabel('Residuals', fontsize=10)
        axes[i, 1].set_ylabel('Frequency', fontsize=10)
        axes[i, 1].set_title(f'Residuals Distribution - {target_name}', fontsize=11)
        axes[i, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/residuals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals analysis to {figures_dir}/residuals_analysis.png")


def main():
    """Main training pipeline."""
    print("Loading data...")
    X, y, feature_names, target_names = load_and_prepare_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of targets: {len(target_names)}")
    print(f"Target statistics:")
    for i, name in enumerate(target_names):
        print(f"  {name}: mean={y[:, i].mean():.2f}, std={y[:, i].std():.2f}")

    # Create figures directory
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    # Load full dataframe for correlation analysis
    print("\nGenerating correlation matrix...")
    df = pd.read_csv('../../data/CVD_cleaned_dummies.csv')
    plot_correlation_matrix(df, target_names, figures_dir)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Standardize features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    # Standardize targets
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Create model
    print("\nCreating model...")
    model = create_multioutput_regression_model(input_dim=X_train.shape[1], output_dim=2)
    model.summary()

    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32)

    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, figures_dir)

    # Evaluate model (inverse transform predictions for interpretability)
    print("\nEvaluating model on test set...")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Use original scale for evaluation
    evaluate_model(model, X_test, y_test, target_names, y_pred=y_pred)

    # Plot predictions and residuals
    print("\nGenerating prediction and residual plots...")
    plot_predictions(y_test, y_pred, target_names, figures_dir)
    plot_residuals(y_test, y_pred, target_names, figures_dir)

    # Save model
    model_path = 'saved_models/height_weight_regression.keras'
    os.makedirs('saved_models', exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save scalers
    import joblib
    scaler_X_path = 'saved_models/height_weight_scaler_X.pkl'
    scaler_y_path = 'saved_models/height_weight_scaler_y.pkl'
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Feature scaler saved to {scaler_X_path}")
    print(f"Target scaler saved to {scaler_y_path}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE - All figures saved to:", figures_dir)
    print("="*50)
    print("Generated figures:")
    print(f"  - {figures_dir}/correlation_matrix_full.png")
    print(f"  - {figures_dir}/correlation_with_targets.png")
    print(f"  - {figures_dir}/training_history.png")
    print(f"  - {figures_dir}/predictions_vs_actual.png")
    print(f"  - {figures_dir}/residuals_analysis.png")

    return model, history


if __name__ == "__main__":
    main()
