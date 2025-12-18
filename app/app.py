import streamlit as st
import numpy as np
from tensorflow import keras
import joblib
import os


@st.cache_resource
def load_models():
    """Load TensorFlow models and scalers."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths
    models_dir = os.path.join(project_root, 'models', 'tensorflow', 'saved_models')
    classification_model_path = os.path.join(models_dir, 'heart_disease_classification.keras')
    regression_model_path = os.path.join(models_dir, 'height_weight_regression.keras')
    classification_scaler_path = os.path.join(models_dir, 'heart_disease_scaler.pkl')
    regression_scaler_X_path = os.path.join(models_dir, 'height_weight_scaler_X.pkl')
    regression_scaler_y_path = os.path.join(models_dir, 'height_weight_scaler_y.pkl')
    threshold_path = os.path.join(models_dir, 'optimal_threshold.txt')

    # Load classification model
    classification_model = keras.models.load_model(classification_model_path)
    classification_scaler = joblib.load(classification_scaler_path)

    # Load regression model
    regression_model = keras.models.load_model(regression_model_path)
    regression_scaler_X = joblib.load(regression_scaler_X_path)
    regression_scaler_y = joblib.load(regression_scaler_y_path)

    # Load optimal threshold
    with open(threshold_path, 'r') as f:
        optimal_threshold = float(f.read().strip())

    return {
        'classification_model': classification_model,
        'classification_scaler': classification_scaler,
        'regression_model': regression_model,
        'regression_scaler_X': regression_scaler_X,
        'regression_scaler_y': regression_scaler_y,
        'optimal_threshold': optimal_threshold
    }


def create_classification_feature_vector(user_data):
    """
    Transform user input into feature vector for classification.
    Order matches CVD_cleaned_dummies.csv columns EXCLUDING Heart_Disease_Yes.
    Total: 37 features
    """
    features = [user_data['Height'], user_data['Weight'], user_data['BMI'], user_data['Alcohol_Consumption'],
                user_data['Fruits_Consumption'], user_data['Vegetables_Consumption'], user_data['Fried_Consumption'],
                1 if user_data['General_Health'] == 'Fair' else 0, 1 if user_data['General_Health'] == 'Good' else 0,
                1 if user_data['General_Health'] == 'Poor' else 0,
                1 if user_data['General_Health'] == 'Very Good' else 0, 1 if user_data['Checkup'] == 'Never' else 0,
                1 if user_data['Checkup'] == 'Within the past 2 years' else 0,
                1 if user_data['Checkup'] == 'Within the past 5 years' else 0,
                1 if user_data['Checkup'] == 'Within the past year' else 0, 1 if user_data['Exercise'] == 'Yes' else 0,
                1 if user_data['Skin_Cancer'] == 'Yes' else 0, 1 if user_data['Other_Cancer'] == 'Yes' else 0,
                1 if user_data['Depression'] == 'Yes' else 0,
                1 if user_data['Diabetes'] == 'No, pre-diabetes or borderline diabetes' else 0,
                1 if user_data['Diabetes'] == 'Yes' else 0,
                1 if user_data['Diabetes'] == 'Yes, but female told only during pregnancy' else 0,
                1 if user_data['Arthritis'] == 'Yes' else 0, 1 if user_data['Sex'] == 'Male' else 0]

    # Age_Category (one-hot: 25-29 through 80+; 18-24 is baseline)
    age_categories = ['25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                      '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    for category in age_categories:
        features.append(1 if user_data['Age_Category'] == category else 0)

    # Smoking_History_Yes
    features.append(1 if user_data['Smoking_History'] == 'Yes' else 0)

    return np.array(features).reshape(1, -1)


def create_regression_feature_vector(user_data, heart_disease_prediction):
    """
    Create feature vector for regression (predicting height and weight).
    Order matches CVD_cleaned_dummies.csv columns EXCLUDING Height_(cm) and Weight_(kg).
    Includes Heart_Disease_Yes as a feature (using the classification prediction).
    Total: 36 features
    """
    features = [user_data['BMI'], user_data['Alcohol_Consumption'], user_data['Fruits_Consumption'],
                user_data['Vegetables_Consumption'], user_data['Fried_Consumption'],
                1 if user_data['General_Health'] == 'Fair' else 0, 1 if user_data['General_Health'] == 'Good' else 0,
                1 if user_data['General_Health'] == 'Poor' else 0,
                1 if user_data['General_Health'] == 'Very Good' else 0, 1 if user_data['Checkup'] == 'Never' else 0,
                1 if user_data['Checkup'] == 'Within the past 2 years' else 0,
                1 if user_data['Checkup'] == 'Within the past 5 years' else 0,
                1 if user_data['Checkup'] == 'Within the past year' else 0, 1 if user_data['Exercise'] == 'Yes' else 0,
                heart_disease_prediction, 1 if user_data['Skin_Cancer'] == 'Yes' else 0,
                1 if user_data['Other_Cancer'] == 'Yes' else 0, 1 if user_data['Depression'] == 'Yes' else 0,
                1 if user_data['Diabetes'] == 'No, pre-diabetes or borderline diabetes' else 0,
                1 if user_data['Diabetes'] == 'Yes' else 0,
                1 if user_data['Diabetes'] == 'Yes, but female told only during pregnancy' else 0,
                1 if user_data['Arthritis'] == 'Yes' else 0, 1 if user_data['Sex'] == 'Male' else 0]
    # Age_Category (one-hot)
    age_categories = ['25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                      '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    for category in age_categories:
        features.append(1 if user_data['Age_Category'] == category else 0)

    # Smoking_History_Yes
    features.append(1 if user_data['Smoking_History'] == 'Yes' else 0)

    return np.array(features).reshape(1, -1)


def display_classification():
    name_field = st.text_input("Enter your name here")
    age_field = st.number_input("Enter your age here", min_value=18, max_value=99)
    gender_field = st.selectbox("Select your gender at birth here", ("Male", "Female"))
    height_field = st.number_input("Enter your height in centimeters here", min_value=75, max_value=250)
    weight_field = st.number_input("Enter your weight in kilograms here", min_value=30, max_value=250)
    health_quality_field = st.selectbox("How would you describe your general health?", ("Excellent",
                                                                                        "Very Good",
                                                                                        "Good",
                                                                                        "Fair",
                                                                                        "Poor",
                                                                                        ))
    last_checkup_field = st.selectbox("When was you last global health checkup?", ('Within the past year',
                                                                                   'Within the past 2 years',
                                                                                   'Within the past 5 years',
                                                                                   '5 or more years ago',
                                                                                   'Never',
                                                                                   ))
    exercise_field = st.selectbox("Do you often exercise?", ("Yes", "No"))
    depression_field = st.selectbox("Are you or have you ever been diagnosed as depressive?", ("No", "Yes"))
    diabetes_field = st.selectbox("Do you have any diabetes?", ("No",
                                                                "No, pre-diabetes or borderline diabetes",
                                                                "Yes",
                                                                f"{"Yes, but female told only during pregnancy" if (
                                                                        gender_field == "Female") else ""}",
                                                                ))

    smoking_field = st.selectbox("Have you ever been a regular smoker at any point of your life?", ("Yes", "No"))
    skin_cancer_field = st.selectbox("Have ever been diagnosed with a skin cancer?", ("No", "Yes"))
    other_cancer_field = st.selectbox("Have ever been diagnosed with any other kind of cancer?", ("No", "Yes"))
    arthritis_field = st.selectbox("Have ever been diagnosed with arthritis?", ("No", "Yes"))

    st.write("")
    st.write("For the next questions you'll have the option to choose if you rather answer as days, weeks, or months.")

    col1, col2 = st.columns(2)

    alcohol_field = col1.number_input("How many times do you drink alcoholic beverages?",
                                      min_value=0, key="achohol_input")
    alcohol_radio = col2.radio("Per:", ("Day", "Week", "Month"), horizontal=True, key="alcohol_radio")

    fruits_field = col1.number_input("How many times do you eat fruits?", min_value=0, key="fruit_input")
    fruits_radio = col2.radio("Per", ("Day", "Week", "Months"), horizontal=True, key="fruit_radio")

    vegetables_field = col1.number_input("How many times do you eat vegetables?", min_value=0, key="vegetables_input")
    vegetables_radio = col2.radio("Per:", ("Day", "Week", "Month"), horizontal=True, key="vegetables_radio")

    fried_field = col1.number_input("How many times do you eat fried potatoes or chicken?",
                                    min_value=0, key="fried_input")
    fried_radio = col2.radio("Per:", ("Day", "Week", "Month"), horizontal=True, key="fried_radio")

    if age_field >= 80:
        age = "80+"
    elif age_field >= 75:
        age = "75-79"
    elif age_field >= 70:
        age = "70-74"
    elif age_field >= 65:
        age = "65-69"
    elif age_field >= 60:
        age = "60-64"
    elif age_field >= 55:
        age = "55-59"
    elif age_field >= 50:
        age = "50-54"
    elif age_field >= 45:
        age = "45-49"
    elif age_field >= 40:
        age = "40-44"
    elif age_field >= 35:
        age = "35-39"
    elif age_field >= 30:
        age = "30-34"
    elif age_field >= 25:
        age = "25-29"
    elif age_field >= 20:
        age = "20-24"
    else:
        age = "18-24"

    bmi = round(weight_field / (height_field ** 2) * 10000, 2)

    alcohol_quantity = alcohol_field
    if alcohol_radio == "Day":
        alcohol_quantity = alcohol_quantity * 30
    elif alcohol_radio == "Week":
        alcohol_quantity = int(alcohol_quantity * 4.2)

    fruits_quantity = fruits_field
    if fruits_radio == "Day":
        fruits_quantity = fruits_quantity * 30
    elif fruits_radio == "Week":
        fruits_quantity = int(fruits_quantity * 4.2)

    vegetables_quantity = vegetables_field
    if vegetables_radio == "Day":
        vegetables_quantity = vegetables_quantity * 30
    elif vegetables_radio == "Week":
        vegetables_quantity = int(vegetables_quantity * 4.2)

    fried_quantity = fried_field
    if fried_radio == "Day":
        fried_quantity = fried_quantity * 30
    elif fried_radio == "Week":
        fried_quantity = int(fried_quantity * 4.2)

    st.divider()

    user_data = {
        "General_Health": health_quality_field,
        "Checkup": last_checkup_field,
        "Exercise": exercise_field,
        "Skin_Cancer": skin_cancer_field,
        "Other_Cancer": other_cancer_field,
        "Depression": depression_field,
        "Diabetes": diabetes_field,
        "Arthritis": arthritis_field,
        "Sex": gender_field,
        "Age_Category": age,
        "Height": height_field,
        "Weight": weight_field,
        "BMI": bmi,
        "Smoking_History": smoking_field,
        "Alcohol_Consumption": alcohol_quantity,
        "Fruits_Consumption": fruits_quantity,
        "Vegetables_Consumption": vegetables_quantity,
        "Fried_Consumption": fried_quantity,
    }

    return user_data, name_field


def main():
    st.set_page_config(page_title="Health Assessment", page_icon="❤️", layout="wide")

    st.title('❤️ Health Self-Assessment Test')
    st.write("This application uses deep learning models to predict heart disease risk and estimate body metrics.")

    st.divider()

    # Load models
    try:
        models = load_models()
        st.success("✓ Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure the models are trained and saved in the correct directory.")
        return

    # Get user input
    user_data, name = display_classification()

    # Predict button
    if st.button("🔍 Analyze My Health", type="primary", use_container_width=True):
        if not name:
            st.warning("Please enter your name to continue.")
            return

        st.divider()
        st.header(f"📊 Results for {name}")

        # Create two columns for results
        col1, col2 = st.columns(2)

        # ===== HEART DISEASE PREDICTION =====
        with col1:
            st.subheader("❤️ Heart Disease Risk Assessment")

            try:
                # Create feature vector for classification
                features = create_classification_feature_vector(user_data)

                # Scale features
                features_scaled = models['classification_scaler'].transform(features)

                # Make prediction
                prediction_proba = float(models['classification_model'].predict(features_scaled, verbose=0)[0][0])

                # Apply optimal threshold
                optimal_threshold = models['optimal_threshold']
                prediction_class = 1 if prediction_proba > optimal_threshold else 0

                # Display results
                st.metric(
                    label="Risk Probability",
                    value=f"{prediction_proba * 100:.1f}%",
                    delta=f"Threshold: {optimal_threshold:.3f}"
                )

                if prediction_class == 1:
                    st.error("⚠️ **HIGH RISK** - Heart disease risk detected")
                    st.warning("**Recommendation:** "
                               "Please consult with a healthcare professional for a comprehensive evaluation.")
                else:
                    st.success("✓ **LOW RISK** - No significant heart disease risk detected")
                    st.info("**Recommendation:** Continue maintaining a healthy lifestyle.")

                # Risk level indicator
                if prediction_proba >= 0.7:
                    risk_level = "Very High"
                    risk_color = "🔴"
                elif prediction_proba >= 0.5:
                    risk_level = "High"
                    risk_color = "🟠"
                elif prediction_proba >= 0.3:
                    risk_level = "Moderate"
                    risk_color = "🟡"
                else:
                    risk_level = "Low"
                    risk_color = "🟢"

                st.write(f"**Risk Level:** {risk_color} {risk_level}")

                # Progress bar for risk (clamp between 0 and 1)
                st.progress(min(1.0, max(0.0, prediction_proba)))

            except Exception as e:
                st.error(f"Error making heart disease prediction: {str(e)}")
                prediction_class = 0  # Default to no heart disease if prediction fails

        # ===== BODY METRICS PREDICTION =====
        with col2:
            st.subheader("📏 Predicted Body Metrics")

            try:
                # Create feature vector for regression (using heart disease prediction)
                features = create_regression_feature_vector(user_data, prediction_class)

                # Scale features
                features_scaled = models['regression_scaler_X'].transform(features)

                # Make prediction
                prediction_scaled = models['regression_model'].predict(features_scaled, verbose=0)

                # Inverse transform to get actual values
                prediction = models['regression_scaler_y'].inverse_transform(prediction_scaled)[0]

                predicted_height = prediction[0]
                predicted_weight = prediction[1]

                # Display results
                st.write("**Predicted vs Actual:**")

                # Height comparison
                height_diff = user_data['Height'] - predicted_height
                st.metric(
                    label="Height (cm)",
                    value=f"{predicted_height:.1f} cm (predicted)",
                    delta=f"{height_diff:.1f} cm difference from actual"
                )

                # Weight comparison
                weight_diff = user_data['Weight'] - predicted_weight
                st.metric(
                    label="Weight (kg)",
                    value=f"{predicted_weight:.1f} kg (predicted)",
                    delta=f"{weight_diff:.1f} kg difference from actual"
                )

                # Actual values
                st.write("**Your Actual Measurements:**")
                st.write(f"- Height: {user_data['Height']:.1f} cm")
                st.write(f"- Weight: {user_data['Weight']:.1f} kg")
                st.write(f"- BMI: {user_data['BMI']:.2f}")

                # BMI interpretation
                bmi = user_data['BMI']
                if bmi < 18.5:
                    bmi_category = "Underweight"
                    bmi_color = "🔵"
                elif bmi < 25:
                    bmi_category = "Normal weight"
                    bmi_color = "🟢"
                elif bmi < 30:
                    bmi_category = "Overweight"
                    bmi_color = "🟡"
                else:
                    bmi_category = "Obese"
                    bmi_color = "🔴"

                st.info(f"**BMI Category:** {bmi_color} {bmi_category}")

            except Exception as e:
                st.error(f"Error making body metrics prediction: {str(e)}")

        # ===== HEALTH SUMMARY =====
        st.divider()
        st.subheader("📋 Health Profile Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.write("**Lifestyle Factors:**")
            st.write(f"- Exercise: {user_data['Exercise']}")
            st.write(f"- Smoking History: {user_data['Smoking_History']}")
            st.write(f"- Alcohol: {user_data['Alcohol_Consumption']}/month")

        with summary_col2:
            st.write("**Medical History:**")
            st.write(f"- Diabetes: {user_data['Diabetes']}")
            st.write(f"- Depression: {user_data['Depression']}")
            st.write(f"- Arthritis: {user_data['Arthritis']}")

        with summary_col3:
            st.write("**Diet & Health:**")
            st.write(f"- General Health: {user_data['General_Health']}")
            st.write(f"- Last Checkup: {user_data['Checkup']}")
            st.write(f"- Fruits: {user_data['Fruits_Consumption']}/month")

        # Disclaimer
        st.divider()
        st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only and should not be used as a"
                   " substitute for professional medical advice."
                   " Always consult with a qualified healthcare provider for medical diagnosis and treatment.")


if __name__ == '__main__':
    main()
