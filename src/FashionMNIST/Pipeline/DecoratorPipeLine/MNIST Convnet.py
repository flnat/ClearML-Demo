# https://keras.io/examples/vision/mnist_convnet/

import numpy as np
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(return_values=["train_data", "train_labels", "test_data", "test_labels"], cache=True,
                             task_type=TaskTypes.data_processing)
def data_retrieval(link: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param link: link to Train-Data, Train-Labels, Test-Data, Test-Labels of the MNIST Dataset
    :return: np.ndarrays of Train-Data, Train-Labels, Test-Data, Test-Labels of the MNIST Dataset
    """
    # Libraries muessen innerhalb der Funktionen importiert werden --> Dependency Discovery des Agents
    import numpy as np
    import tensorflow as tf

    train_data: np.ndarray
    train_labels: np.ndarray
    test_data: np.ndarray
    test_lables: np.ndarray

    (train_data, train_labels), (test_data, test_lables) = tf.keras.datasets.mnist.load_data(path=link)

    return train_data, train_labels, test_data, test_lables


@PipelineDecorator.component(return_values=["train_data", "train_labels", "test_data", "test_labels"], cache=True,
                             task_type=TaskTypes.data_processing)
def data_preparation(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray,
                     test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    
    :return: 
    """
    import numpy as np
    import tensorflow as tf
    num_classes: int = 10
    input_shape: tuple[int, int, int] = (28, 28, 1)

    train_data = train_data.astype("float32") / 255
    test_data = test_data.astype("float32") / 255

    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    return train_data, train_labels, test_data, test_labels


@PipelineDecorator.component(return_values=["fit"], cache=True, task_type=TaskTypes.training)
def model_fit(train_data: np.ndarray, train_labels: np.ndarray, epochs: int, batch_size: int):
    """
    
    :return: 
    """
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Max2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers(10, activation='softmax'))

    model.compile(loss="categorical-crossentropy", optimizer='adam', metrics=["accuracy"])

    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model


@PipelineDecorator.component(return_values=["score"], name="Evaluation", cache=True, task_type=TaskTypes.testing)
def evaluate_fit(model, test_data: np.ndarray, test_labels: np.ndarray):
    score = model.evaluate(test_data, test_labels)

    print("Test Loss: ", score[0])
    print("Test accuracy: ", score[1])
    return score


@PipelineDecorator.pipeline(name="Controller", project="demo", version="0.0.1")
def controller(link: str, epochs: int, batch_size: int) -> None:
    print("Launch task data retrieval")
    train_data, train_labels, test_data, test_labels = data_retrieval(link=link)

    print("Launch task data preparation")
    train_data, train_labels, test_data, test_labels = data_preparation(train_data, train_labels, test_data,
                                                                        test_labels)

    print("Launch task model fitting")
    model = model_fit(train_data, train_labels, epochs, batch_size)

    print("Launch task model evaluation")
    score = evaluate_fit(model, test_data, test_labels)
    print("Pipeline finished")


if __name__ == "__main__":
    # Experiment ohne Agent -->Zeitsparen
    PipelineDecorator.run_locally()
    controller(
        link="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        epochs=15,
        batch_size=128
    )
