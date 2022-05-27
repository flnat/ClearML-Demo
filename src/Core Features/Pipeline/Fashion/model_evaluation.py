import tempfile
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from clearml import Logger, StorageManager, Task, TaskTypes
from tensorflow.keras.callbacks import TensorBoard

# Define Requirements for remote execution by agents
Task.add_requirements("clearml")
Task.add_requirements("numpy")
Task.add_requirements("tensorflow")

# Define helper function for plotting debug samples
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')


task: Task = Task.init(
    project_name="demo/Fashion MNIST",
    task_name="model_evaluation",
    task_type=TaskTypes.testing
)

args: dict[str, typing.Any] = {
    "model_url": "http://localhost:8081/demo/Fashion%20MNIST/model_training.2ea5b7e9aff04bff9dcb006599533bed/models/weight.zip",
    "test_data": "http://localhost:8081/demo/Fashion%20MNIST/data_ingestion.f80d9c1e3dc445d09a7e355840ab8284/artifacts/test_data/test_data.npy",
    "test_labels": "http://localhost:8081/demo/Fashion%20MNIST/data_ingestion.f80d9c1e3dc445d09a7e355840ab8284/artifacts/test_labels/test_labels.npy"
}

task.connect(args)
task.execute_remotely()

logger: Logger = task.get_logger()
manager: StorageManager = StorageManager()

# Get local copies of test data and labels & normalize the data
test_data: np.ndarray = np.load(manager.get_local_copy(remote_url=args["test_data"]))
test_labels: np.ndarray = np.load(manager.get_local_copy(remote_url=args["test_labels"]))

test_data = test_data / 255.0

# Create Tensorboard instance for Evaluation Logging
temp = tempfile.TemporaryDirectory()
tmp_folder: Path = Path(temp.name) / "evaluation"
board = TensorBoard(log_dir=tmp_folder, write_images=True, histogram_freq=1)

# Get a local Copy of the CNN Model and also deserialize it
model = tf.keras.models.load_model(manager.get_local_copy(remote_url=args["model_url"]))
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Evaluate the model and calculate loss and accuracy
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

# Log Test accuracy and loss as text
logger.report_text(
    msg=f"Test accuracy: {test_acc}\n" +
        f"Test loss: {test_loss}"
)

# Log some sample Predictions
predictions = probability_model.predict(test_data)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_data)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()

logger.report_matplotlib_figure(
    title="Sample Predictions",
    series="predictions",
    figure=plt
)
