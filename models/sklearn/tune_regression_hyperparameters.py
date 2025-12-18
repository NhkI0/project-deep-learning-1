import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    df = pd.read_csv(csv_path)
    y = df[['Height_(cm)', 'Weight_(kg)']].values
    X = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).values
    feature_names = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).columns.tolist()
    target_names = ['Height_(cm)', 'Weight_(kg)']
    return X, y, feature_names, target_names


def get_mlp_param_grid(search_type='random'):
    if search_type == 'grid':
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
    else:
        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64),
                                   (128, 64, 32), (256, 128, 64)],
            'activation': ['relu', 'tanh', 'identity'],
            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [100, 200],
            'early_stopping': [True]
        }
    return param_grid


def tune_hyperparameters(X_train, y_train, search_type='random',
                         n_iter=50, cv=3, n_jobs=-1):
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING - MLP - {search_type.upper()} SEARCH")
    print(f"{'='*60}")

    base_model = MLPRegressor(random_state=42, verbose=False, max_iter=100)
    param_grid = get_mlp_param_grid(search_type)

    if search_type == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            scoring='neg_mean_squared_error',
            cv=cv,
            n_jobs=n_jobs,
            verbose=2,
            return_train_score=True
        )
    else:
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=cv,
            n_jobs=n_jobs,
            verbose=2,
            random_state=42,
            return_train_score=True
        )

    print(f"\nStarting hyperparameter search...")
    print(f"Cross-validation folds: {cv}")
    if search_type == 'random':
        print(f"Number of iterations: {n_iter}")
    print(f"This may take a while...\n")

    search.fit(X_train, y_train)

    return search


def evaluate_best_model(search, X_test, y_test, scaler_y, target_names):
    print("\n" + "="*60)
    print("BEST MODEL EVALUATION")
    print("="*60)

    best_params = search.best_params_
    print("\nBest Hyperparameters:")
    print("-" * 40)
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV Score (Negative MSE): {search.best_score_:.4f}")

    best_model = search.best_estimator_

    y_pred_scaled = best_model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 2) if y_pred_scaled.ndim == 1 else y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)

    print("\nTest Set Performance:")

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

    return best_model, best_params


def save_best_model(model, scaler_X, scaler_y, best_params):
    model_path = 'saved_models/best_height_weight_regression_sklearn.pkl'
    scaler_X_path = 'saved_models/best_height_weight_scaler_X_sklearn.pkl'
    scaler_y_path = 'saved_models/best_height_weight_scaler_y_sklearn.pkl'
    params_path = 'saved_models/best_hyperparameters_regression.json'

    os.makedirs('saved_models', exist_ok=True)

    joblib.dump(model, model_path)
    print(f"\nBest model saved to {model_path}")

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Feature scaler saved to {scaler_X_path}")
    print(f"Target scaler saved to {scaler_y_path}")

    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Hyperparameters saved to {params_path}")


def main():
    print("Hyperparameter Tuning for Height & Weight Regression")
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
    y_test = scaler_y.transform(y_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    print("\n" + "="*60)
    print("CHOOSE SEARCH TYPE")
    print("="*60)
    print("1. Random Search (fast, explores broader space)")
    print("2. Grid Search (exhaustive, but slower)")

    search_choice = input("\nEnter search type (1-2) [default: 1]: ").strip() or '1'
    search_types = {'1': 'random', '2': 'grid'}
    search_type = search_types.get(search_choice, 'random')

    n_iter = 30
    if search_type == 'random':
        n_iter_input = input(f"\nNumber of iterations [default: 30]: ").strip()
        n_iter = int(n_iter_input) if n_iter_input else 30

    cv_folds = input(f"\nNumber of CV folds [default: 3]: ").strip()
    cv_folds = int(cv_folds) if cv_folds else 3

    search = tune_hyperparameters(
        X_train, y_train,
        search_type=search_type,
        n_iter=n_iter,
        cv=cv_folds
    )

    best_model, best_params = evaluate_best_model(search, X_test, y_test, scaler_y, target_names)

    save_best_model(best_model, scaler_X, scaler_y, best_params)

    print("\nThe best model has been saved and can be used with:")
    print("  - test_regression_model.py")
    print("  - multioutput_regression_height_weight.py")

    return search, best_model


if __name__ == "__main__":
    main()
