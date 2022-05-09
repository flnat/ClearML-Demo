import gzip
import re
import tempfile
from pathlib import Path

import numpy as np
from clearml import Dataset, Logger

# Inherit from existing Fashion MNIST Dataset

parent: Dataset = Dataset.get(
    dataset_name="Fashion MNIST",
    dataset_project="demo/Fashion MNIST/datasets"
)

dataset: Dataset = Dataset.create(
    dataset_name='Fashion MNIST Matrices',
    dataset_project="demo/Fashion MNIST/datasets",
    dataset_tags=["foreign", "Image Data"],
    parent_datasets=[parent]
)
train_data_pattern: re.Pattern = re.compile(".+\.train-images-idx3-ubyte\.gz")
train_labels_pattern: re.Pattern = re.compile(".+\.train-labels-idx1-ubyte\.gz")
test_data_pattern: re.Pattern = re.compile(".+\.t10k-images-idx3-ubyte\.gz")
test_labels_pattern: re.Pattern = re.compile(".+\.t10k-labels-idx1-ubyte\.gz")

temp_dir = tempfile.TemporaryDirectory()
original_data = Path(dataset.get_local_copy())

# Extraction logic taken from: https://github.com/keras-team/keras/blob/master/keras/datasets/fashion_mnist.py

train_labels_path = next(path for path in original_data.iterdir() if train_labels_pattern.match(path.name))
train_data_path = next(path for path in original_data.iterdir() if train_data_pattern.match(path.name))
test_labels_path = next(path for path in original_data.iterdir() if test_labels_pattern.match(path.name))
test_data_path = next(path for path in original_data.iterdir() if test_data_pattern.match(path.name))

with gzip.open(train_labels_path, "rb") as lblPath:
    train_labels: np.ndarray = np.frombuffer(lblPath.read(), np.uint8, offset=8)

with gzip.open(train_data_path, "rb") as imgPath:
    train_data: np.ndarray = np.frombuffer(
        imgPath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)

with gzip.open(test_labels_path, "rb") as lblPath:
    test_labels: np.ndarray = np.frombuffer(lblPath.read(), np.uint8, offset=8)

with gzip.open(test_data_path, "rb") as imgPath:
    test_data: np.ndarray = np.frombuffer(
        imgPath.read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28)

# with np.load(os.path.join(original_data, os.listdir(original_data)[0]), allow_pickle=True) as data:
#     train_data, train_labels = data['x_train'], data['y_train']
#     test_data, test_labels = data['x_test'], data['y_test']

logger: Logger = dataset.get_logger()

# Clear out old data
dataset.remove_files(dataset_path="*")

# Upload ndarray of data as .npz archives

np.save(file=temp_dir.name + "\\train_data", arr=train_data, allow_pickle=True)
np.save(file=temp_dir.name + "\\train_labels", arr=train_labels, allow_pickle=True)
np.save(file=temp_dir.name + "\\test_data", arr=test_data, allow_pickle=True)
np.save(file=temp_dir.name + "\\test_labels", arr=test_labels, allow_pickle=True)

dataset.add_files(
    path=temp_dir.name,
    wildcard="*.npy"
)

# Upload first 50 training Samples as Pictures
for i in range(50):
    image_data = train_data[i] * 255
    logger.report_image(
        title="Fashion MNIST Items",
        series="item",
        iteration=i,
        image=image_data
    )

dataset.upload()
dataset.finalize()
temp_dir.cleanup()
