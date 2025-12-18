import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


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


def create_mlp_regressor(random_state=42):
    """Create an improved multi-output regression model matching TensorFlow architecture."""
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=True
    )
    return model


def train_model(model, X_train, y_train):
    print("\nTraining model...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names, y_pred=None, model_name="Model"):
    """Evaluate the trained model."""
    if y_pred is None:
        y_pred = model.predict(X_test)

    print("\n" + "="*50)
    print(f"MODEL EVALUATION - {model_name}")
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


def main():
    """Main training pipeline."""
    print("="*60)
    print("HEIGHT & WEIGHT REGRESSION - Scikit-learn")
    print("="*60)

    print("\nLoading data...")
    X, y, feature_names, target_names = load_and_prepare_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of targets: {len(target_names)}")
    print(f"Target statistics:")
    for i, name in enumerate(target_names):
        print(f"  {name}: mean={y[:, i].mean():.2f}, std={y[:, i].std():.2f}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Standardize features
    print("\nStandardizing features...")
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    # Standardize targets
    print("Standardizing targets...")
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Create model
    print("\nCreating model...")
    print("Architecture: 256 -> 128 -> 64 -> 32 -> 16")
    model = create_mlp_regressor()

    # Train model
    model = train_model(model, X_train, y_train)

    # Evaluate model (inverse transform predictions for interpretability)
    print("\nEvaluating model on test set...")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Use original scale for evaluation
    evaluate_model(model, X_test, y_test, target_names, y_pred=y_pred, model_name="MLPRegressor")

    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)

    os.makedirs('saved_models', exist_ok=True)

    model_path = 'saved_models/height_weight_regression_sklearn.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")

    scaler_X_path = 'saved_models/height_weight_scaler_X_sklearn.pkl'
    scaler_y_path = 'saved_models/height_weight_scaler_y_sklearn.pkl'
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"✓ Feature scaler saved to {scaler_X_path}")
    print(f"✓ Target scaler saved to {scaler_y_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nConfiguration (matching TensorFlow/Keras):")
    print(f"  - Architecture: 256 -> 128 -> 64 -> 32 -> 16")
    print(f"  - Learning rate: 0.001")
    print(f"  - Activation: ReLU")
    print(f"  - Early stopping: enabled")

    return model, scaler_X, scaler_y


if __name__ == "__main__":
    main()
