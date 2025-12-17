import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os
import json


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    df = pd.read_csv(csv_path)
    y = df['Heart_Disease_Yes'].values
    X = df.drop('Heart_Disease_Yes', axis=1).values
    feature_names = df.drop('Heart_Disease_Yes', axis=1).columns.tolist()
    return X, y, feature_names


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
            'activation': ['relu', 'tanh', 'logistic'],
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

    base_model = MLPClassifier(random_state=42, verbose=False, max_iter=100)
    param_grid = get_mlp_param_grid(search_type)

    if search_type == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            scoring='roc_auc',
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
            scoring='roc_auc',
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


def evaluate_best_model(search, X_test, y_test):
    print("\n" + "="*60)
    print("BEST MODEL EVALUATION")
    print("="*60)

    best_params = search.best_params_
    print("\nBest Hyperparameters:")
    print("-" * 40)
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV Score (ROC-AUC): {search.best_score_:.4f}")

    best_model = search.best_estimator_

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

    return best_model, best_params


def compare_top_models(search, X_test, y_test, top_n=5):
    print("\n" + "="*60)
    print(f"COMPARING TOP {top_n} MODELS")
    print("="*60)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values('rank_test_score')

    results = []

    for i in range(min(top_n, len(cv_results))):
        params = cv_results.iloc[i]['params']
        cv_score = cv_results.iloc[i]['mean_test_score']

        model = search.estimator.set_params(**params)
        model.fit(search.best_estimator_.n_features_in_ if hasattr(search.best_estimator_, 'n_features_in_') else X_test.shape[1])

        try:
            y_pred_proba = search.estimator.set_params(**params).fit(
                X_test if i == 0 else search.cv_results_['params'][0],
                y_test if i == 0 else y_test
            ).predict_proba(X_test)[:, 1]
            y_pred = search.estimator.set_params(**params).predict(X_test)
        except:
            continue

        results.append({
            'rank': i + 1,
            'cv_score': cv_score,
            'params': params
        })

        print(f"\nModel #{i+1}:")
        print(f"  CV ROC-AUC: {cv_score:.4f}")
        print(f"  Parameters: {params}")

    return results


def save_best_model(model, scaler, best_params):
    model_path = 'saved_models/best_heart_disease_classification_sklearn.pkl'
    scaler_path = 'saved_models/best_heart_disease_scaler_sklearn.pkl'
    params_path = 'saved_models/best_hyperparameters_classification.json'

    os.makedirs('saved_models', exist_ok=True)

    joblib.dump(model, model_path)
    print(f"\nBest model saved to {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Hyperparameters saved to {params_path}")


def main():
    print("Hyperparameter Tuning for Heart Disease Classification")
    print("\nLoading data...")
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

    best_model, best_params = evaluate_best_model(search, X_test, y_test)

    save_best_model(best_model, scaler, best_params)

    print("\nThe best model has been saved and can be used with:")
    print("  - test_classification_model.py")
    print("  - classification_heart_disease.py")

    return search, best_model


if __name__ == "__main__":
    main()
