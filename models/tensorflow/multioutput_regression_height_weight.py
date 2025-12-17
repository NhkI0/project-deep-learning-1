import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def create_multioutput_regression_model(input_dim, output_dim=2):
    """Create a multi-output regression model for Height and Weight prediction."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(output_dim)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the multi-output regression model."""
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


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate the trained model."""
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
    y_test_scaled = scaler_y.transform(y_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Create model
    print("\nCreating model...")
    model = create_multioutput_regression_model(input_dim=X_train.shape[1], output_dim=2)
    model.summary()

    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

    # Evaluate model (inverse transform predictions for interpretability)
    print("\nEvaluating model on test set...")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Use original scale for evaluation
    temp_model_output = y_pred
    evaluate_model(
        lambda x: temp_model_output if x is X_test else model.predict(x),
        X_test,
        y_test,
        target_names
    )

    # Save model
    model_path = 'saved_models/height_weight_regression'
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

    return model, history


if __name__ == "__main__":
    main()
