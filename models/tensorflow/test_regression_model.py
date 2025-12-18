import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import os


def load_model_and_scalers():
    model_path = 'saved_models/height_weight_regression.keras'
    scaler_X_path = 'saved_models/height_weight_scaler_X.pkl'
    scaler_y_path = 'saved_models/height_weight_scaler_y.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Feature scaler not found at {scaler_X_path}. Please train the model first.")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Target scaler not found at {scaler_y_path}. Please train the model first.")

    model = keras.models.load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_X, scaler_y


def get_feature_names():
    df = pd.read_csv('../../data/CVD_cleaned_dummies.csv')
    feature_names = df.drop(['Height_(cm)', 'Weight_(kg)'], axis=1).columns.tolist()
    return feature_names


def create_sample_input():
    feature_names = get_feature_names()

    sample = {}

    sample['BMI'] = 24.22
    sample['Alcohol_Consumption'] = 0.0
    sample['Fruit_Consumption'] = 16.0
    sample['Green_Vegetables_Consumption'] = 8.0
    sample['FriedPotato_Consumption'] = 4.0

    for feature in feature_names:
        if feature not in sample:
            sample[feature] = False

    return sample


def predict_from_dict(model, scaler_X, scaler_y, input_dict, feature_names):
    input_array = np.array([[input_dict[feature] for feature in feature_names]])

    input_scaled = scaler_X.transform(input_array)

    prediction_scaled = model.predict(input_scaled, verbose=0)

    prediction = scaler_y.inverse_transform(prediction_scaled)[0]

    height_pred = prediction[0]
    weight_pred = prediction[1]

    return height_pred, weight_pred


def explain_prediction(height_pred, weight_pred, input_dict):
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)

    print(f"\nPredicted Height: {height_pred:.2f} cm")
    print(f"Predicted Weight: {weight_pred:.2f} kg")

    height_m = height_pred / 100
    predicted_bmi = weight_pred / (height_m ** 2)
    print(f"Predicted BMI (from outputs): {predicted_bmi:.2f}")

    if 'BMI' in input_dict:
        input_bmi = input_dict['BMI']
        print(f"Input BMI: {input_bmi:.2f}")
        print(f"Difference: {abs(predicted_bmi - input_bmi):.2f}")

    print("\n" + "-"*60)
    print("HOW TO INTERPRET:")
    print("-"*60)

    print("\nThe model predicts height and weight based on:")
    print("  - Health indicators (BMI, diseases, conditions)")
    print("  - Lifestyle factors (exercise, diet, smoking)")
    print("  - Demographics (age, sex)")

    print("\nHeight Categories (cm):")
    if height_pred < 160:
        height_cat = "Below average"
    elif height_pred < 175:
        height_cat = "Average"
    else:
        height_cat = "Above average"
    print(f"  Predicted height is: {height_cat}")

    print("\nWeight Categories (kg):")
    if weight_pred < 60:
        weight_cat = "Below average"
    elif weight_pred < 80:
        weight_cat = "Average"
    else:
        weight_cat = "Above average"
    print(f"  Predicted weight is: {weight_cat}")

    print("\nBMI Categories:")
    if predicted_bmi < 18.5:
        bmi_cat = "Underweight"
    elif predicted_bmi < 25:
        bmi_cat = "Normal weight"
    elif predicted_bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"
    print(f"  Predicted BMI indicates: {bmi_cat}")


def interactive_mode():
    print("\n" + "="*60)
    print("INTERACTIVE TESTING MODE - Height & Weight Prediction")
    print("="*60)

    print("\nLoading model and scalers...")
    model, scaler_X, scaler_y = load_model_and_scalers()
    feature_names = get_feature_names()

    print(f"\nModel loaded successfully!")
    print(f"Number of input features: {len(feature_names)}")
    print("Output: Height (cm) and Weight (kg)")

    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
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

            height_pred, weight_pred = predict_from_dict(
                model, scaler_X, scaler_y, sample, feature_names
            )
            explain_prediction(height_pred, weight_pred, sample)

        elif choice == '2':
            sample = create_sample_input()
            print("\nStarting with sample values. You can modify them.")

            print("\n--- Continuous Features ---")
            try:
                bmi = float(input(f"BMI [default: {sample['BMI']:.2f}]: ") or sample['BMI'])
                alcohol = float(input(f"Alcohol Consumption [default: {sample['Alcohol_Consumption']:.1f}]: ") or sample['Alcohol_Consumption'])
                fruit = float(input(f"Fruit Consumption [default: {sample['Fruit_Consumption']:.1f}]: ") or sample['Fruit_Consumption'])
                veggies = float(input(f"Green Vegetables Consumption [default: {sample['Green_Vegetables_Consumption']:.1f}]: ") or sample['Green_Vegetables_Consumption'])
                potatoes = float(input(f"Fried Potato Consumption [default: {sample['FriedPotato_Consumption']:.1f}]: ") or sample['FriedPotato_Consumption'])

                sample['BMI'] = bmi
                sample['Alcohol_Consumption'] = alcohol
                sample['Fruit_Consumption'] = fruit
                sample['Green_Vegetables_Consumption'] = veggies
                sample['FriedPotato_Consumption'] = potatoes

                print("\n--- Boolean Features (enter 1 for True, 0 for False) ---")
                print("(Press Enter to keep default False)")

                important_features = [
                    'Heart_Disease_Yes', 'Exercise_Yes', 'Diabetes_Yes',
                    'Arthritis_Yes', 'Sex_Male', 'Smoking_History_Yes',
                    'Depression_Yes'
                ]

                for feature in important_features:
                    if feature in feature_names:
                        value = input(f"{feature} [0/1, default: 0]: ").strip()
                        sample[feature] = bool(int(value)) if value else False

                print("\n--- Age Category (only one should be True) ---")
                age_categories = [f for f in feature_names if f.startswith('Age_Category_')]
                print("Available age categories:")
                for i, cat in enumerate(age_categories, 1):
                    print(f"  {i}. {cat}")
                age_choice = input("Select age category number (or press Enter for none): ").strip()
                if age_choice and age_choice.isdigit():
                    idx = int(age_choice) - 1
                    if 0 <= idx < len(age_categories):
                        sample[age_categories[idx]] = True

            except ValueError:
                print("\nInvalid input. Using default sample values.")
                sample = create_sample_input()

            print("\nYour custom input:")
            print_input_summary(sample)

            height_pred, weight_pred = predict_from_dict(
                model, scaler_X, scaler_y, sample, feature_names
            )
            explain_prediction(height_pred, weight_pred, sample)

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
    continuous = ['BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
                  'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
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
    model, scaler_X, scaler_y = load_model_and_scalers()
    feature_names = get_feature_names()

    custom_input = create_sample_input()

    custom_input['BMI'] = 28.5
    custom_input['Exercise_Yes'] = True
    custom_input['Sex_Male'] = True
    custom_input['Age_Category_40-44'] = True
    custom_input['Smoking_History_Yes'] = False

    print("\nCustom input:")
    print_input_summary(custom_input)

    height_pred, weight_pred = predict_from_dict(
        model, scaler_X, scaler_y, custom_input, feature_names
    )
    explain_prediction(height_pred, weight_pred, custom_input)


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
