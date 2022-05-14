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

# Initialize the Task --> Start ClearML Integration
task: Task = Task.init(
    project_name="demo/Fashion MNIST",
    task_name="model_training",
    task_type=TaskTypes.training,
    output_uri=True
)

# Dict of Hyperparameters with default values for ease of debugging
args: dict[str, typing.Any] = {
    "train_data": "http://localhost:8081/demo/Fashion%20MNIST/data_ingestion.f80d9c1e3dc445d09a7e355840ab8284/artifacts/train_data/train_data.npy",
    "train_labels": "http://localhost:8081/demo/Fashion%20MNIST/data_ingestion.f80d9c1e3dc445d09a7e355840ab8284/artifacts/train_labels/train_labels.npy",
    "epochs": 5,
    "batch_size": 256,
    "layer_1": 64
}
task.connect(args)
classes: tuple[str, ...] = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")
enumeration = {k: v for v, k in enumerate(classes, 1)}
Task.current_task().connect_label_enumeration(enumeration)
# task.execute_remotely()

# Initiate Logger for some later logging
logger: Logger = task.get_logger()
manager: StorageManager = StorageManager()

train_data: np.ndarray = np.load(manager.get_local_copy(remote_url=args["train_data"]), allow_pickle=True)
train_labels: np.ndarray = np.load(manager.get_local_copy(remote_url=args["train_labels"]), allow_pickle=True)

# Data Normalization
train_data = train_data / 255.0

# Model definition

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(args["layer_1"], activation='relu'),
    keras.layers.Dense(10)
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
