from clearml import Dataset, StorageManager

manager: StorageManager = StorageManager()

dataset_path = StorageManager.get_local_copy(
    remote_url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
)
dataset: Dataset = Dataset.create(
    dataset_name="MNIST",
    dataset_project="demo/Fashion MNIST/datasets",
    dataset_tags=["zipped", "foreign", "Image Data"]
)

dataset.add_files(
    path=dataset_path
)
dataset.upload()
dataset.finalize()
