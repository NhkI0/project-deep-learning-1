import pandas as pd


def load_linear_regression():
    df = pd.read_csv("data/CVD_cleaned_dummies.csv")
    cols_to_remove = ["Height_(cm)", "Weight_(kg)"]
    return df.drop(columns=cols_to_remove)


def load_classification():
    df = pd.read_csv("data/CVD_cleaned_dummies.csv")
    cols_to_remove = ["Heart_Disease_Yes"]
    return df.drop(columns=cols_to_remove)
