from clearml import Dataset, StorageManager

manager: StorageManager = StorageManager()

dataset: Dataset = Dataset.create(
    dataset_name="Fashion MNIST",
    dataset_project="demo/Fashion MNIST/datasets",
    dataset_tags=["zipped", "foreign", "Image Data", "not ready"]
)

BASE_URL: str = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
files: list[str] = [
    'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
]

for file in files:
    dataset_path = StorageManager.get_local_copy(remote_url=BASE_URL + file)
    dataset.add_files(path=dataset_path)


dataset.upload()
dataset.finalize()
