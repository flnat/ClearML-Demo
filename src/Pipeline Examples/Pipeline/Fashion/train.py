import typing
import os

from clearml import Task, TaskTypes, Logger, OutputModel, StorageManager
import tensorflow as tf
import numpy as np
import tempfile

# Provide requirements for the Script
Task.add_requirements("numpy")
Task.add_requirements("tensorflow")
Task.add_requirements("clearml")

task: Task = Task.init(project_name="demo/Fashion MNIST", task_name="model_training", task_type=TaskTypes.training)

args: dict[str, typing.Any] = {
    "train_data": "http://localhost:8081/demo/Fashion%20MNIST/.pipelines/Fashion%20MNIST%20Pipeline/data_ingestion.48789a8667904a6596eac7ee490987f6/artifacts/train_data/train_data.npy",
    "train_labels": "http://localhost:8081/demo/Fashion%20MNIST/.pipelines/Fashion%20MNIST%20Pipeline/data_ingestion.48789a8667904a6596eac7ee490987f6/artifacts/train_labels/train_labels.npy",
    "test_data": "http://localhost:8081/demo/Fashion%20MNIST/.pipelines/Fashion%20MNIST%20Pipeline/data_ingestion.48789a8667904a6596eac7ee490987f6/artifacts/test_data/test_data.npy",
    "test_labels": "http://localhost:8081/demo/Fashion%20MNIST/.pipelines/Fashion%20MNIST%20Pipeline/data_ingestion.48789a8667904a6596eac7ee490987f6/artifacts/test_labels/test_labels.npy",
    "epochs": 10,
    "batch_size": 256
}
task.connect(args)

# task.execute_remotely()

# Initiate Logger for some later logging
logger: Logger = task.get_logger()
manager: StorageManager = StorageManager()

train_data: np.ndarray = np.load(manager.get_local_copy(remote_url=args["train_data"]), allow_pickle=True)
train_labels: np.ndarray = np.load(manager.get_local_copy(remote_url=args["train_labels"]), allow_pickle=True)
test_data: np.ndarray = np.load(manager.get_local_copy(remote_url=args["test_data"]), allow_pickle=True)
test_labels: np.ndarray = np.load(manager.get_local_copy(remote_url=args["test_labels"]), allow_pickle=True)

# Data Normalization
train_data = train_data / 255.0

# Model definition

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Take a look at the model summary
model.summary()

logger.report_text(
    msg=model.summary(),
    print_console=False
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


results = model.fit(
    train_data, train_labels,
    epochs=args["epochs"],
    batch_size=args["batch_size"]
)

history = results.history

for idx, acc in enumerate(history["accuracy"]):
    logger.report_scalar(
        title="Model Accuracy",
        series="series",
        value=acc,
        iteration=idx + 1
    )

for idx, loss in enumerate(history["loss"]):
    logger.report_scalar(
        title="Model Loss",
        series="series",
        value=loss,
        iteration=idx + 1
    )


