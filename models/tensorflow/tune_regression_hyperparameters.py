import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    df = pd.read_csv(csv_path)
    y = df[['Height_(cm)', 'Weight_(kg)']].values
    X = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).values
    feature_names = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).columns.tolist()
    target_names = ['Height_(cm)', 'Weight_(kg)']
    return X, y, feature_names, target_names


def build_model(hp, input_dim):

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    num_layers = hp.Int('num_layers', min_value=2, max_value=5, step=1)

    activation = hp.Choice('activation', values=['relu', 'elu', 'selu', 'tanh'])

    for i in range(num_layers):
        units = hp.Choice(f'units_layer_{i}', values=[32, 64, 128, 256])
        model.add(keras.layers.Dense(units, activation=activation))

        dropout_rate = hp.Float(f'dropout_layer_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(2))

    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def tune_hyperparameters(X_train, y_train, X_val, y_val, input_dim,
                         max_trials=50, executions_per_trial=1, search_type='random'):
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING - {search_type.upper()} SEARCH")
    print(f"{'='*60}")

    if search_type == 'random':
        tuner = kt.RandomSearch(
            hypermodel=lambda hp: build_model(hp, input_dim),
            objective=kt.Objective('val_loss', direction='min'),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuning_results',
            project_name='regression_random_search',
            overwrite=True
        )
    elif search_type == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: build_model(hp, input_dim),
            objective=kt.Objective('val_loss', direction='min'),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuning_results',
            project_name='regression_bayesian',
            overwrite=True
        )
    elif search_type == 'hyperband':
        tuner = kt.Hyperband(
            hypermodel=lambda hp: build_model(hp, input_dim),
            objective=kt.Objective('val_loss', direction='min'),
            max_epochs=100,
            factor=3,
            directory='tuning_results',
            project_name='regression_hyperband',
            overwrite=True
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    tuner.search_space_summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print(f"\nStarting hyperparameter search with {max_trials} trials...")
    print(f"This may take a while depending on your hardware...\n")

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    return tuner


def evaluate_best_model(tuner, X_test, y_test, scaler_y, target_names):
    print("\n" + "="*60)
    print("BEST MODEL EVALUATION")
    print("="*60)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest Hyperparameters:")
    print("-" * 40)
    for param, value in best_hps.values.items():
        if param != 'input_dim':
            print(f"  {param}: {value}")

    best_model = tuner.get_best_models(num_models=1)[0]

    y_pred_scaled = best_model.predict(X_test, verbose=0)

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)

    print("Evaluating best model")

    for i, target_name in enumerate(target_names):
        y_true_i = y_test_original[:, i]
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

    overall_mse = mean_squared_error(y_test_original, y_pred)
    overall_mae = mean_absolute_error(y_test_original, y_pred)

    print(f"\nOverall:")
    print(f"  MSE: {overall_mse:.4f}")
    print(f"  MAE: {overall_mae:.4f}")

    return best_model, best_hps


def compare_top_models(tuner, X_test, y_test, scaler_y, target_names, top_n=5):
    print("\n" + "="*60)
    print(f"Comparing {top_n} best models")

    top_hps = tuner.get_best_hyperparameters(num_trials=top_n)
    top_models = tuner.get_best_models(num_models=top_n)

    results = []

    for i, (model, hps) in enumerate(zip(top_models, top_hps), 1):
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test)

        overall_mse = mean_squared_error(y_test_original, y_pred)
        overall_mae = mean_absolute_error(y_test_original, y_pred)

        r2_scores = []
        for j in range(len(target_names)):
            r2 = r2_score(y_test_original[:, j], y_pred[:, j])
            r2_scores.append(r2)

        results.append({
            'rank': i,
            'mse': overall_mse,
            'mae': overall_mae,
            'r2_height': r2_scores[0],
            'r2_weight': r2_scores[1],
            'hyperparameters': {k: v for k, v in hps.values.items() if k != 'input_dim'}
        })

        print(f"\nModel #{i}:")
        print(f"  MSE: {overall_mse:.4f}")
        print(f"  MAE: {overall_mae:.4f}")
        print(f"  R² (Height): {r2_scores[0]:.4f}")
        print(f"  R² (Weight): {r2_scores[1]:.4f}")

    return results


def save_best_model(model, scaler_X, scaler_y, best_hps):
    model_path = 'saved_models/best_height_weight_regression'
    scaler_X_path = 'saved_models/best_height_weight_scaler_X.pkl'
    scaler_y_path = 'saved_models/best_height_weight_scaler_y.pkl'
    hps_path = 'saved_models/best_hyperparameters_regression.json'

    os.makedirs('saved_models', exist_ok=True)

    model.save(model_path)
    print(f"\nBest model saved to {model_path}")

    import joblib
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Feature scaler saved to {scaler_X_path}")
    print(f"Target scaler saved to {scaler_y_path}")

    hps_dict = {k: v for k, v in best_hps.values.items() if k != 'input_dim'}
    with open(hps_path, 'w') as f:
        json.dump(hps_dict, f, indent=2)
    print(f"Hyperparameters saved to {hps_path}")


def main():
    print("Hyperparameters tuning:")
    print("\nLoading data...")
    X, y, feature_names, target_names = load_and_prepare_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of targets: {len(target_names)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    print("  1. Random Search (fast, good for initial exploration)")
    print("  2. Bayesian Optimization (smart, learns from previous trials)")
    print("  3. Hyperband (efficient, adaptive resource allocation)")

    choice = input("\nChoose search type (1-3) [default: 1]: ").strip() or '1'

    search_types = {'1': 'random', '2': 'bayesian', '3': 'hyperband'}
    search_type = search_types.get(choice, 'random')

    max_trials = input("\nNumber of trials to run [default: 30]: ").strip()
    max_trials = int(max_trials) if max_trials else 30

    tuner = tune_hyperparameters(
        X_train, y_train, X_val, y_val,
        input_dim=X_train.shape[1],
        max_trials=max_trials,
        executions_per_trial=1,
        search_type=search_type
    )

    best_model, best_hps = evaluate_best_model(tuner, X_test, y_test, scaler_y, target_names)

    results = compare_top_models(tuner, X_test, y_test, scaler_y, target_names, top_n=5)

    save_best_model(best_model, scaler_X, scaler_y, best_hps)

    print("\nThe best model has been saved and can be used with:")
    print("  - test_regression_model.py")
    print("  - multioutput_regression_height_weight.py")

    return tuner, best_model, results

if __name__ == "__main__":
    main()
