import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath, ',')


def drop_all_na(dataframe):
    return pd.DataFrame.dropna(dataframe)