from pathlib import Path

import pandas as pd
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(return_values=["pandas DataFrame"], task_type=TaskTypes.data_processing)
def data_cleanup() -> pd.DataFrame:
    import os
    import pandas as pd
    from clearml import Dataset

    dataset: Dataset = Dataset.get(
        dataset_name="Airbnb NYC",
        dataset_project="demo/Airbnb"
    )

    dataset_path: str = dataset.get_local_copy()
    file: str = os.path.join(
        dataset_path,
        os.listdir(dataset_path)[0]
    )
    df: pd.DataFrame = pd.read_csv(file)

    df.drop_duplicates(inplace=True)
    df.drop(['name', 'id', 'host_name', 'host_id', 'last_review', 'license'], axis=1, inplace=True)
    df.fillna({'reviews_per_month': 0}, inplace=True)
    df.dropna(how='any', inplace=True)
    return df


@PipelineDecorator.component(return_values=["dummy"], task_type=TaskTypes.monitor)
def describe_data(df: pd.DataFrame) -> bool:
    from clearml import Logger, Task
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger: Logger = Task.current_task().get_logger()

    logger.report_table(
        title="Summary of continuous Variables",
        series="Table",
        table_plot=df.describe()
    )

    # Correlation Matrix
    corr = df.corr(method="kendall")
    sns.heatmap(corr, annot=True)

    logger.report_matplotlib_figure(
        title="Correlation Matrix",
        series="Heatmap",
        figure=plt,
        report_interactive=False
    )
    # Plot Neighbourhood_groups
    sns.scatterplot(df.longitude, df.latitude, hue=df.neighbourhood_group)
    logger.report_matplotlib_figure(
        title="Neighbourhood Group",
        series="Scatterplot",
        figure=plt,
        report_interactive=False
    )
    plt.show()

    # Location + Type Plot
    sns.scatterplot(df.longitude, df.latitude, hue=df.room_type)

    logger.report_matplotlib_figure(
        title="Location-RoomType Mapping",
        series="Scatterplot",
        figure=plt,
        report_interactive=False
    )
    plt.show()
    # Availability
    sns.boxplot(data=df, x='neighbourhood_group', y='availability_365', palette="plasma")

    logger.report_matplotlib_figure(
        title="Availability per Neighbourhood",
        series="Boxplot",
        figure=plt,
        report_interactive=False
    )
    plt.show()
    return True


@PipelineDecorator.component(return_values=["encoded_dataFrame"], task_type=TaskTypes.data_processing)
def encoding(df: pd.DataFrame) -> pd.DataFrame:
    from clearml import Logger
    df_en = df.copy()
    df_en.drop(
        ['latitude', 'longitude', 'neighbourhood', 'number_of_reviews', 'reviews_per_month', 'number_of_reviews_ltm'],
        axis=1, inplace=True
    )

    for column in df_en.columns[df_en.columns.isin(['neighbourhood_group', 'room_type'])]:
        df_en[column] = df_en[column].factorize()[0]

    logger: Logger = Logger.current_logger()
    logger.report_table(
        title="Encoded DataFrame",
        series="pandas DataFrame",
        table_plot=df_en
    )

    return df_en


@PipelineDecorator.component(return_values=["model"], task_type=TaskTypes.training)
def linear_regression(df: pd.DataFrame):
    from clearml import Logger
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import joblib

    x = df[df.columns[~df.columns.isin(['price'])]]
    y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=353)

    # Prepare Linear Regression
    reg: LinearRegression = LinearRegression()
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    joblib.dump(reg, "linearRegression_model.pkl", compress=True)
    score = r2_score(y_test, y_predict)
    logger = Logger.current_logger()
    logger.report_text(msg=f"R2-Score:{score}")

    logger.report_scalar(
        "R2",
        "R2",
        score,
        0
    )

    return reg


@PipelineDecorator.component(return_values=['model'], task_type=TaskTypes.training)
def decisiontree_regression(df: pd.DataFrame):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from clearml import Logger
    import joblib
    x = df[df.columns[~df.columns.isin(['price'])]]
    y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=353)
    DTree = DecisionTreeRegressor(min_samples_leaf=.0001)
    DTree.fit(x_train, y_train)
    y_predict = DTree.predict(x_test)
    joblib.dump(DTree, "decisionTree_model.pkl", compress=True)

    score = r2_score(y_test, y_predict)

    logger = Logger.current_logger()
    logger.report_text(msg=f"R2-Score:{score}")

    logger.report_scalar(
        "R2",
        "R2",
        score,
        0
    )

    return DTree


@PipelineDecorator.pipeline(name="AirBNB NYC Pipeline", project="demo/Airbnb", version="0.0.1")
def executing_pipeline() -> None:
    df: pandas.DataFrame = data_cleanup()
    describe_data(df)
    df_en: pandas.DataFrame = encoding(df)
    reg = linear_regression(df=df_en)
    tree = decisiontree_regression(df=df_en)


if __name__ == '__main__':
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    # PipelineDecorator.set_default_execution_queue('default')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()
    # Start the pipeline execution logic.
    executing_pipeline()

    print('process completed')
    # Remove .pkl artifacts in cwd

    cwd = Path.cwd()
    for item in cwd.iterdir():
        if item.suffix == ".pkl":
            item.unlink()
