import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import os


def load_model_and_scaler():
    model_path = 'saved_models/heart_disease_classification.keras'
    scaler_path = 'saved_models/heart_disease_scaler.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def get_feature_names():
    df = pd.read_csv('../../data/CVD_cleaned_dummies.csv')
    feature_names = df.drop('Heart_Disease_Yes', axis=1).columns.tolist()
    return feature_names


def create_sample_input():
    feature_names = get_feature_names()

    sample = {}

    sample['Height_(cm)'] = 170.0
    sample['Weight_(kg)'] = 70.0
    sample['BMI'] = 24.22
    sample['Alcohol_Consumption'] = 0.0
    sample['Fruit_Consumption'] = 16.0
    sample['Green_Vegetables_Consumption'] = 8.0
    sample['FriedPotato_Consumption'] = 4.0

    for feature in feature_names:
        if feature not in sample:
            sample[feature] = False

    return sample


def predict_from_dict(model, scaler, input_dict, feature_names):
    input_array = np.array([[input_dict[feature] for feature in feature_names]])

    input_scaled = scaler.transform(input_array)

    prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
    prediction_class = int(prediction_proba > 0.5)

    return prediction_proba, prediction_class


def explain_prediction(prediction_proba, prediction_class):
    print(f"\nPrediction Probability: {prediction_proba:.4f} ({prediction_proba*100:.2f}%)")
    print(f"Predicted Class: {prediction_class} ({'Heart Disease' if prediction_class == 1 else 'No Heart Disease'})")

def interactive_mode():
    print("\n" + "="*60)
    print("INTERACTIVE TESTING MODE - Heart Disease Prediction")
    print("="*60)

    print("\nLoading model and scaler...")
    model, scaler = load_model_and_scaler()
    feature_names = get_feature_names()

    print(f"\nModel loaded successfully!")
    print(f"Number of features: {len(feature_names)}")

    while True:
        print("Options:")
        print("  1. Test with sample input")
        print("  2. Test with custom input (modify sample)")
        print("  3. View all feature names")
        print("  4. Exit")
        print("-"*60)

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            sample = create_sample_input()
            print("\nUsing sample input:")
            print_input_summary(sample)

            prediction_proba, prediction_class = predict_from_dict(
                model, scaler, sample, feature_names
            )
            explain_prediction(prediction_proba, prediction_class)

        elif choice == '2':
            sample = create_sample_input()
            print("\nStarting with sample values. You can modify them.")

            print("\n--- Continuous Features ---")
            try:
                height = float(input(f"Height (cm) [default: {sample['Height_(cm)']:.1f}]: ") or sample['Height_(cm)'])
                weight = float(input(f"Weight (kg) [default: {sample['Weight_(kg)']:.1f}]: ") or sample['Weight_(kg)'])
                bmi = float(input(f"BMI [default: {sample['BMI']:.2f}]: ") or sample['BMI'])
                alcohol = float(input(f"Alcohol Consumption [default: {sample['Alcohol_Consumption']:.1f}]: ") or sample['Alcohol_Consumption'])
                fruit = float(input(f"Fruit Consumption [default: {sample['Fruit_Consumption']:.1f}]: ") or sample['Fruit_Consumption'])
                veggies = float(input(f"Green Vegetables Consumption [default: {sample['Green_Vegetables_Consumption']:.1f}]: ") or sample['Green_Vegetables_Consumption'])
                potatoes = float(input(f"Fried Potato Consumption [default: {sample['FriedPotato_Consumption']:.1f}]: ") or sample['FriedPotato_Consumption'])

                sample['Height_(cm)'] = height
                sample['Weight_(kg)'] = weight
                sample['BMI'] = bmi
                sample['Alcohol_Consumption'] = alcohol
                sample['Fruit_Consumption'] = fruit
                sample['Green_Vegetables_Consumption'] = veggies
                sample['FriedPotato_Consumption'] = potatoes

                print("\n--- Boolean Features (enter 1 for True, 0 for False) ---")
                print("(Press Enter to keep default False)")

                important_features = [
                    'Exercise_Yes', 'Skin_Cancer_Yes', 'Other_Cancer_Yes',
                    'Depression_Yes', 'Diabetes_Yes', 'Arthritis_Yes',
                    'Sex_Male', 'Smoking_History_Yes'
                ]

                for feature in important_features:
                    if feature in feature_names:
                        value = input(f"{feature} [0/1, default: 0]: ").strip()
                        sample[feature] = bool(int(value)) if value else False

            except ValueError:
                print("\nInvalid input. Using default sample values.")
                sample = create_sample_input()

            print("\nYour custom input:")
            print_input_summary(sample)

            prediction_proba, prediction_class = predict_from_dict(
                model, scaler, sample, feature_names
            )
            explain_prediction(prediction_proba, prediction_class)

        elif choice == '3':
            print("\nAll feature names:")
            for i, feature in enumerate(feature_names, 1):
                print(f"  {i:2d}. {feature}")

        elif choice == '4':
            print("\nExiting. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter 1-4.")


def print_input_summary(input_dict):
    print("\nContinuous features:")
    continuous = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
                  'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
    for feature in continuous:
        if feature in input_dict:
            print(f"  {feature}: {input_dict[feature]}")

    print("\nActive boolean features (True):")
    true_features = [k for k, v in input_dict.items() if k not in continuous and v]
    if true_features:
        for feature in true_features:
            print(f"  {feature}")
    else:
        print("  (None)")


def test_with_custom_dict():
    print("\n" + "="*60)
    print("EXAMPLE: Testing with custom dictionary")
    print("="*60)

    model, scaler = load_model_and_scaler()
    feature_names = get_feature_names()

    custom_input = create_sample_input()

    custom_input['Height_(cm)'] = 165.0
    custom_input['Weight_(kg)'] = 85.0
    custom_input['BMI'] = 31.2
    custom_input['Exercise_Yes'] = False
    custom_input['Diabetes_Yes'] = True
    custom_input['Sex_Male'] = True
    custom_input['Age_Category_60-64'] = True
    custom_input['Smoking_History_Yes'] = True

    print("\nCustom input:")
    print_input_summary(custom_input)

    prediction_proba, prediction_class = predict_from_dict(
        model, scaler, custom_input, feature_names
    )
    explain_prediction(prediction_proba, prediction_class)


if __name__ == "__main__":
    print("  1. Interactive mode (recommended)")
    print("  2. Test with example dictionary")

    mode = input("\nEnter mode (1 or 2): ").strip()

    if mode == '1':
        interactive_mode()
    elif mode == '2':
        test_with_custom_dict()
    else:
        print("\nInvalid choice. Running interactive mode by default.")
        interactive_mode()
