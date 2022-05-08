from clearml import StorageManager, Dataset

#https://www.tensorflow.org/tutorials/keras/classification
def upload_data() -> None:
    """
    Download the Fashion MNIST Data Set and upload it to the ClearMLServer
    :return: None
    """
    download_links: list[str] = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # Train-Data
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",  # Train-Labels
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # Test-Data
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"  # Test-Labels
    ]
    manager: StorageManager = StorageManager()
    dataset: Dataset = Dataset.create(
        dataset_name="Fashion MNIST",
        dataset_project="Demo"
    )

    for link in download_links:
        dataset_path: str = manager.get_local_copy(remote_url=link)
        dataset.add_files(path=dataset_path)

    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    upload_data()