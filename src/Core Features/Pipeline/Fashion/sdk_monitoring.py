from clearml import Task

task_list: list[Task] = Task.get_tasks(
    project_name="demo/Fashion MNIST/.pipelines/Fashion MNIST Pipeline",
    task_name="model_training",
    task_filter={
        "status": ["completed"]
    }
)

task_performance: dict[str: dict] = {}

for task in task_list:
    try:
        epoch_accuracy = task.get_last_scalar_metrics()["epoch_accuracy"]["epoch_accuracy"]
        task_performance[task.id] = epoch_accuracy["max"]
    except KeyError:
        continue

# Print the Tasks ID with their correspondent accuracy to stdOut
for key in sorted(task_performance, key=task_performance.get, reverse=True):
    print(f"{key}: {task_performance[key]}")

# Give new best Task a fitting label
best_task = Task.get_task(
    task_id=max(task_performance, key=task_performance.get)
)

best_task.add_tags("best")
