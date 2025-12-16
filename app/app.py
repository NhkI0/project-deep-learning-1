import streamlit as st


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
    last_checkup_field = st.selectbox("When was you last gloabl health checkup?", ('Within the past year',
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

    return {"General_Health": health_quality_field,
            "Checkup": last_checkup_field,
            "Exercise": exercise_field,
            "Skin_Cancer": skin_cancer_field,
            "Other_Cancer": other_cancer_field,
            "Depression": depression_field,
            "Diabetes": diabetes_field,
            "Arthritis": arthritis_field,
            "Sex": gender_field,
            "Age_Category": age,
            "BMI": bmi,
            "Smoking_History": smoking_field,
            "Alcohol_Consumption": alcohol_quantity,
            "Fruits_Consumption": fruits_quantity,
            "Vegetables_Consumption": vegetables_quantity,
            "Fried_Consumption": fried_quantity,
            }


def main():
    st.title('Disease Self-Assessment Test')

    st.write(display_classification())


if __name__ == '__main__':
    main()
