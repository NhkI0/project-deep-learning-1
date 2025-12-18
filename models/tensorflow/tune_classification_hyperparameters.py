import numpy as np
import pandas as pd
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import os
import json


def load_and_prepare_data(csv_path='../../data/CVD_cleaned_dummies.csv'):
    df = pd.read_csv(csv_path)
    y = df['Heart_Disease_Yes'].values
    X = df.drop('Heart_Disease_Yes', axis=1).values
    feature_names = df.drop('Heart_Disease_Yes', axis=1).columns.tolist()
    return X, y, feature_names


def build_model(hp, input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    num_layers = hp.Int('num_layers', min_value=2, max_value=5, step=1)

    activation = hp.Choice('activation', values=['relu', 'elu', 'selu'])

    for i in range(num_layers):
        units = hp.Choice(f'units_layer_{i}', values=[32, 64, 128, 256])
        model.add(keras.layers.Dense(units, activation=activation))

        dropout_rate = hp.Float(f'dropout_layer_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
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
            objective=kt.Objective('val_auc', direction='max'),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuning_results',
            project_name='classification_random_search',
            overwrite=True
        )
    elif search_type == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: build_model(hp, input_dim),
            objective=kt.Objective('val_auc', direction='max'),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuning_results',
            project_name='classification_bayesian',
            overwrite=True
        )
    elif search_type == 'hyperband':
        tuner = kt.Hyperband(
            hypermodel=lambda hp: build_model(hp, input_dim),
            objective=kt.Objective('val_auc', direction='max'),
            max_epochs=100,
            factor=3,
            directory='tuning_results',
            project_name='classification_hyperband',
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


def evaluate_best_model(tuner, X_test, y_test):
    print("Evaluting best model...")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest Hyperparameters:")
    for param, value in best_hps.values.items():
        if param != 'input_dim':
            print(f"  {param}: {value}")

    best_model = tuner.get_best_models(num_models=1)[0]

    y_pred_proba = best_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

    return best_model, best_hps


def compare_top_models(tuner, X_test, y_test, top_n=5):
    print(f"COmparing {top_n} best models")

    top_hps = tuner.get_best_hyperparameters(num_trials=top_n)
    top_models = tuner.get_best_models(num_models=top_n)

    results = []

    for i, (model, hps) in enumerate(zip(top_models, top_hps), 1):
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'rank': i,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'hyperparameters': {k: v for k, v in hps.values.items() if k != 'input_dim'}
        })

        print(f"\nModel #{i}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

    return results


def save_best_model(model, scaler, best_hps):
    model_path = 'saved_models/best_heart_disease_classification.keras'
    scaler_path = 'saved_models/best_heart_disease_scaler.pkl'
    hps_path = 'saved_models/best_hyperparameters_classification.json'

    os.makedirs('saved_models', exist_ok=True)

    model.save(model_path)
    print(f"\nBest model saved to {model_path}")

    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    hps_dict = {k: v for k, v in best_hps.values.items() if k != 'input_dim'}
    with open(hps_path, 'w') as f:
        json.dump(hps_dict, f, indent=2)
    print(f"Hyperparameters saved to {hps_path}")


def main():
    print("Hyperparameter tuning")
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

    best_model, best_hps = evaluate_best_model(tuner, X_test, y_test)

    results = compare_top_models(tuner, X_test, y_test, top_n=5)

    save_best_model(best_model, scaler, best_hps)

    print("\nThe best model has been saved and can be used with:")
    print("  - test_classification_model.py")
    print("  - classification_heart_disease.py")

    return tuner, best_model, results


if __name__ == "__main__":
    main()
