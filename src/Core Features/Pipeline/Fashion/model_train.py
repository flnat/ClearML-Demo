import tempfile
import typing
from pathlib import Path

import numpy as np
from clearml import Logger, StorageManager, Task, TaskTypes
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


# Provide requirements for the Script
Task.add_requirements("numpy")
Task.add_requirements("tensorflow")
Task.add_requirements("clearml")

task: Task = Task.init(project_name="demo/Fashion MNIST", task_name="model_training", task_type=TaskTypes.training,
                       output_uri=True)

args: dict[str, typing.Any] = {
    "train_data": "http://localhost:8081/demo/Fashion%20MNIST/data_ingestion.f80d9c1e3dc445d09a7e355840ab8284/artifacts/train_data/train_data.npy",
    "train_labels": "http://localhost:8081/demo/Fashion%20MNIST/data_ingestion.f80d9c1e3dc445d09a7e355840ab8284/artifacts/train_labels/train_labels.npy",
    # "test_data": "http://localhost:8081/demo/Fashion%20MNIST/.pipelines/Fashion%20MNIST%20Pipeline/data_ingestion.48789a8667904a6596eac7ee490987f6/artifacts/test_data/test_data.npy",
    # "test_labels": "http://localhost:8081/demo/Fashion%20MNIST/.pipelines/Fashion%20MNIST%20Pipeline/data_ingestion.48789a8667904a6596eac7ee490987f6/artifacts/test_labels/test_labels.npy",
    "epochs": 5,
    "batch_size": 256
}
task.connect(args)

# task.execute_remotely()

# Initiate Logger for some later logging
logger: Logger = task.get_logger()
manager: StorageManager = StorageManager()

train_data: np.ndarray = np.load(manager.get_local_copy(remote_url=args["train_data"]), allow_pickle=True)
train_labels: np.ndarray = np.load(manager.get_local_copy(remote_url=args["train_labels"]), allow_pickle=True)
# test_data: np.ndarray = np.load(manager.get_local_copy(remote_url=args["test_data"]), allow_pickle=True)
# test_labels: np.ndarray = np.load(manager.get_local_copy(remote_url=args["test_labels"]), allow_pickle=True)

# Data Normalization
train_data = train_data / 255.0

# Model definition

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (2, 2), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
# Log the Model summary to the console

logger.report_text(
    msg=model.summary(),
    print_console=False
)

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
temp = tempfile.TemporaryDirectory()
tmp_folder: Path = Path(temp.name) / "model"

# Initiate automatic Keras logging with Tensorboard and Callbacks
board = TensorBoard(log_dir=tmp_folder, write_images=True, histogram_freq=1)
model_store = ModelCheckpoint(filepath=tmp_folder / "weight", monitor="accuracy", mode="max", save_best_only=True)

results = model.fit(
    train_data, train_labels,
    epochs=args["epochs"],
    batch_size=args["batch_size"],
    callbacks=[board, model_store]
)

# Log the Accuracy and Loss of the Model Iterations as scalars
# history = results.history
#
# for idx, acc in enumerate(history["accuracy"]):
#     logger.report_scalar(
#         title="Model Accuracy",
#         series="series",
#         value=acc,
#         iteration=idx + 1
#     )
#
# for idx, loss in enumerate(history["loss"]):
#     logger.report_scalar(
#         title="Model Loss",
#         series="series",
#         value=loss,
#         iteration=idx + 1
#     )

# Save the Model to file
# temp = tempfile.TemporaryDirectory()
# tmp_folder: Path = Path(temp.name) / "model"
# model.save(tmp_folder)
