import typing

import clearml.automation.optimization
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch, UniformParameterRange
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

args: dict[str, typing.Any] = {
    "template_task_id": None,
    "run_as_service": False
}

task.connect(args)

execution_queue: str = "default"

optimizer: HyperParameterOptimizer = HyperParameterOptimizer(
    base_task_id=args["template_task_id"],
    hyper_parameters=[
        DiscreteParameterRange("General/batch_size", values=[96, 128, 160]),
        DiscreteParameterRange("General/batch_size", values=[20])
    ],
    objective_metric_title="epoch_accuracy",
    objective_metric_series="epoch_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,
    optimizer_class=search_strategy,
    execution_queue=execution_queue,
    spawn_project=None,
    save_top_k_tasks_only=5,
    time_limit_per_job=5,
    pool_period_min=0.2,
    total_max_jobs=10,
    max_iteration_per_job=30
)

optimizer.set_report_period(2.2)
optimizer.start(job_complete_callback=job_complete_callback)
