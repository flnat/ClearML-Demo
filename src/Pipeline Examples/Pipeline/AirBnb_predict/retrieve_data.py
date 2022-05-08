from clearml import StorageManager, Dataset, Logger

manager: StorageManager = StorageManager()

URL: str = "http://data.insideairbnb.com/united-states/ny/new-york-city/2021-12-04/visualisations/listings.csv"

file: str = StorageManager.get_local_copy(remote_url=URL)

dataset: Dataset = Dataset.create(
    dataset_name="Airbnb NYC",
    dataset_project="demo/Airbnb",
    dataset_tags=["Not zipped", "Not ready"]
)

dataset.add_files(file)
logger: Logger = dataset.get_logger()

logger.report_table(
    title="Airbnb in NYC 2021 Listing",
    series="table",
    url=file
)

dataset.upload()
dataset.finalize()
