import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


def retrieve_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    data.drop(['name', 'id', 'host_name', 'host_id', 'last_review', 'license'], axis=1, inplace=True)
    data.fillna({'reviews_per_month': 0}, inplace=True)
    data.dropna(how='any', inplace=True)
    return data


def describe_data(df: pd.DataFrame) -> bool:
    # Summary Statistics of Continuous Variables
    df.describe()
    corr = df.corr(method="kendall")
    sns.heatmap(corr, annot=True)
    plt.show()
    return True


def encoding(df: pd.DataFrame) -> pd.DataFrame:
    df_en = df.copy()
    df_en.drop(
        ['latitude', 'longitude', 'neighbourhood', 'number_of_reviews', 'reviews_per_month', 'number_of_reviews_ltm'],
        axis=1, inplace=True
    )

    for column in df_en.columns[df_en.columns.isin(['neighbourhood_group', 'room_type'])]:
        df_en[column] = df_en[column].factorize()[0]
    return df_en


def linear_regression(df: pd.DataFrame):
    x = df[df.columns[~df.columns.isin(['price'])]]
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=353)
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    return r2_score(y_test, y_pred)


def decision_tree_regression(df: pd.DataFrame):
    x = df[df.columns[~df.columns.isin(['price'])]]
    y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=353)
    DTree = DecisionTreeRegressor(min_samples_leaf=.0001)
    DTree.fit(x_train, y_train)
    y_pred = DTree.predict(x_test)

    return r2_score(y_test, y_pred)


DATASET_URL = "http://data.insideairbnb.com/united-states/ny/new-york-city/2021-12-04/visualisations/listings.csv"
data = retrieve_data(DATASET_URL)

describe_data(data)

data = prep_data(data)
data = encoding(data)

print(f"Linear Regression R2-Score:{linear_regression(data)}")
print(f"DecisionTree Regression R2-Score:{decision_tree_regression(data)}")
