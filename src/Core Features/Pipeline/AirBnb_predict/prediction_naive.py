import pandas as pd
import matplotlib.pyplot as plt


def retrieve_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    data.drop(['name', 'id', 'host_name', 'host_id', 'last_review', 'license'], axis=1, inplace=True)
    data.fillna({'reviews_per_month': 0}, inplace=True)
    data.dropna(how='any', inplace=True)
    return data


def exploratory_analysis(data: pd.DataFrame) -> pd.DataFrame:
