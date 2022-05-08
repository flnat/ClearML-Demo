import os
import tempfile

import numpy as np
from clearml import Dataset, Logger

# Inherit from existing MNIST Dataset

parent: Dataset = Dataset.get(
    dataset_name="MNIST",
    dataset_project="demo/Fashion MNIST/datasets"
)

dataset: Dataset = Dataset.create(
    dataset_name='MNIST Matrices',
    dataset_project="demo/Fashion MNIST/datasets",
    dataset_tags=["foreign", "Image Data"],
    parent_datasets=[parent]
)

temp_dir = tempfile.TemporaryDirectory()
original_data = dataset.get_local_copy()

train_data: np.ndarray
train_labels: np.ndarray
test_data: np.ndarray
test_labels: np.ndarray

with np.load(os.path.join(original_data, os.listdir(original_data)[0]), allow_pickle=True) as data:
    train_data, train_labels = data['x_train'], data['y_train']
    test_data, test_labels = data['x_test'], data['y_test']

logger: Logger = dataset.get_logger()

# Delete old data
print(dataset.remove_files(dataset_path="*.npz"))

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
        title="MNIST Number",
        series="Number",
        iteration=i,
        image=image_data
    )

dataset.upload()
dataset.finalize()
temp_dir.cleanup()
