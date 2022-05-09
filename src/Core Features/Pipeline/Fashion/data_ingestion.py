import typing

from clearml import Task, TaskTypes, Dataset
from pathlib import Path
import numpy as np

Task.add_requirements("clearml")
Task.add_requirements("numpy")

task: Task = Task.init(project_name="demo/Fashion MNIST", task_name="data_ingestion",
                       task_type=TaskTypes.data_processing)

args: dict[str, typing.Any] = {
    "dataset_name": "Fashion MNIST Matrices",
    "dataset_project": "demo/Fashion MNIST/datasets"
}

task.connect(args)

task.execute_remotely()

dataset: Dataset = Dataset.get(
    dataset_name=args["dataset_name"],
    dataset_project=args["dataset_project"]
)

path: Path = Path(dataset.get_local_copy())

# Register dataset contents as artifact
for file in path.iterdir():
    if file.is_file():
        task.upload_artifact(
            name=file.stem,
            artifact_object=file,
        )
