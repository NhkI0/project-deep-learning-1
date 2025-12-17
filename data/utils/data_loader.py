import pandas as pd


def load_data():
    df = pd.read_csv("data/CVD_cleaned_dummies.csv")
    return df
