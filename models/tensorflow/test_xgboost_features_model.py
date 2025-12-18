import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import os


def engineer_features(data):
    """Apply the same feature engineering as training - XGBoost notebook approach."""
    data = data.copy()

    # 1. BMI Category (categorical labels)
    data['BMI_Category'] = pd.cut(
        data['BMI'],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    )

    # 2. Checkup Frequency
    checkup_mapping = {
        'Within the past year': 4,
        'Within the past 2 years': 2,
        'Within the past 5 years': 1,
        '5 or more years ago': 0.2,
        'Never': 0
    }
    data['Checkup_Frequency'] = data['Checkup'].replace(checkup_mapping)

    # 3. Lifestyle Score
    exercise_mapping = {'Yes': 1, 'No': 0}
    smoking_mapping = {'Yes': -1, 'No': 0}

    data['Lifestyle_Score'] = (
        data['Exercise'].replace(exercise_mapping)
        - data['Smoking_History'].replace(smoking_mapping)
        + data['Fruit_Consumption']/10
        + data['Green_Vegetables_Consumption']/10
        - data['Alcohol_Consumption']/10
    )

    # 4. Healthy Diet Score
    data['Healthy_Diet_Score'] = (
        data['Fruit_Consumption']/10
        + data['Green_Vegetables_Consumption']/10
        - data['FriedPotato_Consumption']/10
    )

    # 5. Interaction Terms
    data['Smoking_Alcohol'] = (
        data['Smoking_History'].replace(smoking_mapping) * data['Alcohol_Consumption']
    )

    data['Checkup_Exercise'] = (
        data['Checkup_Frequency'] * data['Exercise'].replace(exercise_mapping)
    )

    # 6. Height to Weight Ratio
    data['Height_to_Weight'] = data['Height_(cm)'] / data['Weight_(kg)']

    # 7. Fruit and Vegetables Interaction
    data['Fruit_Vegetables'] = (
        data['Fruit_Consumption'] * data['Green_Vegetables_Consumption']
    )

    # 8. Healthy Diet × Lifestyle Interaction
    data['HealthyDiet_Lifestyle'] = (
        data['Healthy_Diet_Score'] * data['Lifestyle_Score']
    )

    # 9. Alcohol × Fried Potato Interaction
    data['Alcohol_FriedPotato'] = (
        data['Alcohol_Consumption'] * data['FriedPotato_Consumption']
    )

    return data


def preprocess_data(data):
    """Apply the same preprocessing as training - XGBoost notebook approach."""
    data = data.copy()

    # Diabetes mapping
    diabetes_mapping = {
        'No': 0,
        'No, pre-diabetes or borderline diabetes': 0,
        'Yes, but female told only during pregnancy': 1,
        'Yes': 1
    }
    data['Diabetes'] = data['Diabetes'].map(diabetes_mapping)

    # One-hot encoding for Sex
    data = pd.get_dummies(data, columns=['Sex'])

    # Binary columns
    binary_columns = [
        'Heart_Disease', 'Skin_Cancer', 'Other_Cancer',
        'Depression', 'Arthritis', 'Smoking_History', 'Exercise'
    ]
    for column in binary_columns:
        if column in data.columns:
            data[column] = data[column].map({'Yes': 1, 'No': 0})

    # Ordinal encoding for General_Health
    general_health_mapping = {
        'Poor': 0,
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Excellent': 4
    }
    data['General_Health'] = data['General_Health'].map(general_health_mapping)

    # Ordinal encoding for BMI_Category
    bmi_mapping = {
        'Underweight': 0,
        'Normal weight': 1,
        'Overweight': 2,
        'Obesity': 3
    }
    data['BMI_Category'] = data['BMI_Category'].map(bmi_mapping).astype(int)

    # Ordinal encoding for Age_Category
    age_category_mapping = {
        '18-24': 0,
        '25-29': 1,
        '30-34': 2,
        '35-39': 3,
        '40-44': 4,
        '45-49': 5,
        '50-54': 6,
        '55-59': 7,
        '60-64': 8,
        '65-69': 9,
        '70-74': 10,
        '75-79': 11,
        '80+': 12
    }
    data['Age_Category'] = data['Age_Category'].map(age_category_mapping)

    # Drop Checkup column
    data = data.drop(["Checkup"], axis=1)

    return data


def load_model_and_scaler():
    """Load model, scaler, and optimal threshold."""
    model_path = 'saved_models/heart_disease_xgboost_features.keras'
    scaler_path = 'saved_models/heart_disease_xgboost_scaler.pkl'
    threshold_path = 'saved_models/optimal_threshold_xgboost.txt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Load optimal threshold
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            optimal_threshold = float(f.read().strip())
    else:
        print("⚠️  Warning: Optimal threshold not found. Using default 0.5")
        optimal_threshold = 0.5

    return model, scaler, optimal_threshold


def get_sample_data():
    """Get sample data from CSV for testing."""
    df = pd.read_csv('../../data/CVD_cleaned.csv')
    return df


def create_sample_input():
    """Create a sample input dictionary for testing."""
    sample = {
        'General_Health': 'Good',
        'Checkup': 'Within the past year',
        'Exercise': 'Yes',
        'Skin_Cancer': 'No',
        'Other_Cancer': 'No',
        'Depression': 'No',
        'Diabetes': 'No',
        'Arthritis': 'No',
        'Sex': 'Male',
        'Age_Category': '50-54',
        'Height_(cm)': 175.0,
        'Weight_(kg)': 80.0,
        'BMI': 26.1,
        'Smoking_History': 'No',
        'Alcohol_Consumption': 2.0,
        'Fruit_Consumption': 20.0,
        'Green_Vegetables_Consumption': 15.0,
        'FriedPotato_Consumption': 4.0
    }
    return sample


def predict_from_dict(model, scaler, input_dict, threshold=0.5):
    """Make prediction from dictionary input."""
    # Create DataFrame from input
    df = pd.DataFrame([input_dict])

    # Apply feature engineering
    df = engineer_features(df)

    # Apply preprocessing
    df = preprocess_data(df)

    # Drop Heart_Disease if it exists
    if 'Heart_Disease' in df.columns:
        df = df.drop('Heart_Disease', axis=1)

    # Ensure both Sex columns exist (one-hot encoding issue)
    if 'Sex_Male' not in df.columns:
        df['Sex_Male'] = 0
    if 'Sex_Female' not in df.columns:
        df['Sex_Female'] = 0

    # Load expected feature names and reorder columns
    features_path = 'saved_models/xgboost_feature_names.txt'
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            expected_features = [line.strip() for line in f.readlines()]

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match training
        df = df[expected_features]

    # Convert to numpy array
    input_array = df.values

    # Scale
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
    prediction_class = int(prediction_proba > threshold)

    return prediction_proba, prediction_class


def explain_prediction(prediction_proba, prediction_class, threshold=0.5):
    """Explain prediction with threshold information."""
    print("\n" + "="*60)
    print("PREDICTION RESULT (XGBoost Features Model)")
    print("="*60)

    print(f"\nPrediction Probability: {prediction_proba:.4f} ({prediction_proba*100:.2f}%)")
    print(f"Decision Threshold: {threshold:.3f} (calibrated for 85% recall)")
    print(f"Predicted Class: {prediction_class} ({'Heart Disease' if prediction_class == 1 else 'No Heart Disease'})")

    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)

    if prediction_proba >= 0.7:
        confidence = "HIGH"
        interpretation = "Strong indication of heart disease risk."
    elif prediction_proba >= threshold:
        confidence = "MEDIUM-HIGH"
        interpretation = "Above screening threshold - medical evaluation recommended."
    elif prediction_proba >= 0.3:
        confidence = "MEDIUM"
        interpretation = "Moderate risk - monitor and consider lifestyle changes."
    else:
        confidence = "LOW"
        interpretation = "Low risk based on current features."

    print(f"Risk Level: {confidence}")
    print(f"{interpretation}")

    print("\n" + "-"*60)
    print("MODEL FEATURES")
    print("-"*60)
    print("This model uses 47+ features including:")
    print("  - Original features (height, weight, BMI, etc.)")
    print("  - Engineered features (lifestyle scores, interactions)")
    print("  - BMI categories, checkup frequency")
    print("  - Health × lifestyle interactions")


def interactive_mode():
    """Interactive mode for testing multiple inputs."""
    print("\n" + "="*60)
    print("INTERACTIVE TESTING - XGBoost Features Model")
    print("="*60)

    print("\nLoading model, scaler, and optimal threshold...")
    model, scaler, optimal_threshold = load_model_and_scaler()

    print(f"\n✓ Model loaded successfully!")
    print(f"✓ Optimal threshold: {optimal_threshold:.3f} (calibrated for 85% recall)")
    print(f"✓ Features: 47+ (with engineered features)")

    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("  1. Test with sample input")
        print("  2. Test with custom input")
        print("  3. Exit")
        print("-"*60)

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            sample = create_sample_input()
            print("\nUsing sample input:")
            print_input_summary(sample)

            prediction_proba, prediction_class = predict_from_dict(
                model, scaler, sample, threshold=optimal_threshold
            )
            explain_prediction(prediction_proba, prediction_class, threshold=optimal_threshold)

        elif choice == '2':
            sample = create_sample_input()
            print("\nStarting with sample values. You can modify them.")

            try:
                # Basic features
                print("\n--- Basic Information ---")
                sample['General_Health'] = input(f"General Health (Poor/Fair/Good/Very Good/Excellent) [default: {sample['General_Health']}]: ").strip() or sample['General_Health']
                sample['Age_Category'] = input(f"Age Category (18-24, 25-29, ..., 80+) [default: {sample['Age_Category']}]: ").strip() or sample['Age_Category']
                sample['Sex'] = input(f"Sex (Male/Female) [default: {sample['Sex']}]: ").strip() or sample['Sex']

                # Physical measurements
                print("\n--- Physical Measurements ---")
                height = input(f"Height (cm) [default: {sample['Height_(cm)']:.1f}]: ").strip()
                if height:
                    sample['Height_(cm)'] = float(height)

                weight = input(f"Weight (kg) [default: {sample['Weight_(kg)']:.1f}]: ").strip()
                if weight:
                    sample['Weight_(kg)'] = float(weight)

                bmi = input(f"BMI [default: {sample['BMI']:.2f}]: ").strip()
                if bmi:
                    sample['BMI'] = float(bmi)

                # Lifestyle factors
                print("\n--- Lifestyle Factors ---")
                sample['Exercise'] = input(f"Exercise (Yes/No) [default: {sample['Exercise']}]: ").strip() or sample['Exercise']
                sample['Smoking_History'] = input(f"Smoking History (Yes/No) [default: {sample['Smoking_History']}]: ").strip() or sample['Smoking_History']

                alcohol = input(f"Alcohol Consumption [default: {sample['Alcohol_Consumption']:.1f}]: ").strip()
                if alcohol:
                    sample['Alcohol_Consumption'] = float(alcohol)

                fruit = input(f"Fruit Consumption [default: {sample['Fruit_Consumption']:.1f}]: ").strip()
                if fruit:
                    sample['Fruit_Consumption'] = float(fruit)

                veggies = input(f"Green Vegetables Consumption [default: {sample['Green_Vegetables_Consumption']:.1f}]: ").strip()
                if veggies:
                    sample['Green_Vegetables_Consumption'] = float(veggies)

                potatoes = input(f"Fried Potato Consumption [default: {sample['FriedPotato_Consumption']:.1f}]: ").strip()
                if potatoes:
                    sample['FriedPotato_Consumption'] = float(potatoes)

            except ValueError:
                print("\nInvalid input. Using default sample values.")
                sample = create_sample_input()

            print("\nYour custom input:")
            print_input_summary(sample)

            prediction_proba, prediction_class = predict_from_dict(
                model, scaler, sample, threshold=optimal_threshold
            )
            explain_prediction(prediction_proba, prediction_class, threshold=optimal_threshold)

        elif choice == '3':
            print("\nExiting. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter 1-3.")


def print_input_summary(input_dict):
    """Print a summary of the input values."""
    print("\nInput Summary:")
    print(f"  General Health: {input_dict['General_Health']}")
    print(f"  Age: {input_dict['Age_Category']}")
    print(f"  Sex: {input_dict['Sex']}")
    print(f"  Height: {input_dict['Height_(cm)']} cm")
    print(f"  Weight: {input_dict['Weight_(kg)']} kg")
    print(f"  BMI: {input_dict['BMI']}")
    print(f"  Exercise: {input_dict['Exercise']}")
    print(f"  Smoking: {input_dict['Smoking_History']}")
    print(f"  Alcohol: {input_dict['Alcohol_Consumption']}")
    print(f"  Fruits: {input_dict['Fruit_Consumption']}")
    print(f"  Vegetables: {input_dict['Green_Vegetables_Consumption']}")


def test_with_custom_dict():
    """Example: Test with a custom dictionary."""
    print("\n" + "="*60)
    print("EXAMPLE: Testing with custom dictionary")
    print("="*60)

    model, scaler, optimal_threshold = load_model_and_scaler()

    # Create custom high-risk profile
    custom_input = {
        'General_Health': 'Fair',
        'Checkup': 'Within the past 2 years',
        'Exercise': 'No',
        'Skin_Cancer': 'No',
        'Other_Cancer': 'No',
        'Depression': 'Yes',
        'Diabetes': 'Yes',
        'Arthritis': 'Yes',
        'Sex': 'Male',
        'Age_Category': '65-69',
        'Height_(cm)': 170.0,
        'Weight_(kg)': 95.0,
        'BMI': 32.9,
        'Smoking_History': 'Yes',
        'Alcohol_Consumption': 10.0,
        'Fruit_Consumption': 5.0,
        'Green_Vegetables_Consumption': 3.0,
        'FriedPotato_Consumption': 16.0
    }

    print("\nCustom input (high-risk profile):")
    print_input_summary(custom_input)

    prediction_proba, prediction_class = predict_from_dict(
        model, scaler, custom_input, threshold=optimal_threshold
    )
    explain_prediction(prediction_proba, prediction_class, threshold=optimal_threshold)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTION - XGBoost Features Model")
    print("="*60)
    print("\nThis model uses extensive feature engineering:")
    print("  - BMI categories, lifestyle scores")
    print("  - Interaction terms (smoking×alcohol, diet×exercise)")
    print("  - 47+ features total")
    print("\nMake sure you have trained the model first by running:")
    print("  python classification_xgboost_features.py")

    print("\n\nChoose a mode:")
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
