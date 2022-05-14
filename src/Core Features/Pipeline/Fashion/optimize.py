import typing

import clearml.automation.optimization
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch, UniformParameterRange, UniformIntegerParameterRange
)

try:
    from clearml.automation.optuna import OptimizerOptuna

    search_strategy = OptimizerOptuna
except ImportError as ex:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB

        search_strategy = OptimizerBOHB
    except ImportError as ex:
        search_strategy = RandomSearch


def job_complete_callback(
        job_id: str,
        objective_value: float,
        objective_iteration: int,
        job_params: dict,
        top_performance_job_id: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('Objective reached {}'.format(objective_value))


task: Task = Task.init(
    project_name="demo/Fashion MNIST",
    task_name="HyperParameter_Optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Find the latest base_task model_training with tag: "draft"-->(Task.TaskStatusEnum.created) for easy debugging
debug_task: Task = Task.get_task(
    # task_id="89009687835e46b8ae89a0bd9919925e"
    project_name="demo/Fashion MNIST",
    task_name="model_training",
    task_filter={"status": ["completed"]}

)

args: dict[str, typing.Any] = {
    "template_task_id": debug_task.id,
    "run_as_service": False
}

task.connect(args)
task.execute_remotely()
execution_queue: str = "default"

optimizer: HyperParameterOptimizer = HyperParameterOptimizer(
    base_task_id=args["template_task_id"],
    hyper_parameters=[
        DiscreteParameterRange("General/batch_size", values=[96, 128, 160]),
        DiscreteParameterRange("General/epochs", values=[number for number in range(5, 16, 5)]),
        UniformIntegerParameterRange("General/layer_1", min_value=64, max_value=256, step_size=64)
    ],
    objective_metric_title="epoch_accuracy",
    objective_metric_series="epoch_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=1,
    optimizer_class=search_strategy,
    execution_queue=execution_queue,
    spawn_project=None,
    save_top_k_tasks_only=1,
    time_limit_per_job=5,
    pool_period_min=0.2,
    total_max_jobs=5,
    max_iteration_per_job=30,
    always_create_task=True
)

optimizer.set_report_period(1.0)
# optimizer.start_locally(job_complete_callback=job_complete_callback)
optimizer.start(job_complete_callback=job_complete_callback)

optimizer.wait()
top_exp = optimizer.get_top_experiments(top_k=1)

task.get_logger().report_text(
    msg=f"ID of Top Experiment: {top_exp}"
)




optimizer.stop()
