import typing

from clearml import Model, OutputModel, Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch, UniformIntegerParameterRange
)

# Example for Optimizer Selection --> in our case always Optuna
# --> other optimizers should behave as drop in replacement
try:
    from clearml.automation.optuna import OptimizerOptuna

    search_strategy = OptimizerOptuna
except ImportError as ex:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB

        search_strategy = OptimizerBOHB
    except ImportError as ex:
        search_strategy = RandomSearch


# Basic Implementation of Optimizer Callback Funtion --> Reporting/Logging
def job_complete_callback(
        job_id: str,
        objective_value: float,
        objective_iteration: int,
        job_parameters: dict,
        top_performance_job_id: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('Objective reached {}'.format(objective_value))


# Task Definition
task: Task = Task.init(
    project_name="demo/Fashion MNIST",
    task_name="HyperParameter_Optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Find the latest base_task model_training with status: "completed"
debug_task: Task = Task.get_task(
    # task_id="89009687835e46b8ae89a0bd9919925e"
    project_name="demo/Fashion MNIST",
    task_name="model_training",
    task_filter={"status": ["completed"]}

)
# Config dict with some default values
args: dict[str, typing.Any] = {
    "template_task_id": debug_task.id,
    "run_as_service": False,
    "max_no_jobs": 10
}

task.connect(args)
# task.execute_remotely()
execution_queue: str = "default"

optimizer: HyperParameterOptimizer = HyperParameterOptimizer(
    base_task_id=args["template_task_id"],
    hyper_parameters=[
        # Discrete Parameter Range for batch_size & epochs
        DiscreteParameterRange("General/batch_size", values=[96, 128, 160]),
        DiscreteParameterRange("General/epochs", values=[number for number in range(5, 16, 5)]),
        # Uniform Int Range for Units in the Model layers
        UniformIntegerParameterRange("General/layer_1_units", min_value=32, max_value=128, step_size=32),
        UniformIntegerParameterRange("General/layer_2_units", min_value=32, max_value=128, step_size=32)
    ],
    # Target Metric -->epoch_accuracy (from TensorBoard Reporting)
    objective_metric_title="epoch_accuracy",
    objective_metric_series="epoch_accuracy",
    # Maximize Metric <> "min"
    objective_metric_sign="max",
    # Do not use multithreading
    max_number_of_concurrent_tasks=1,
    optimizer_class=search_strategy,
    execution_queue=execution_queue,
    spawn_project=None,
    # Keep only the top performing Task
    save_top_k_tasks_only=1,
    time_limit_per_job=5,
    pool_period_min=0.2,
    # Max number of Trials/Tasks/Experiments --> Budget
    total_max_jobs=args["max_no_jobs"],
    max_iteration_per_job=30
)
# Callback accumulated objective report every 60s
optimizer.set_report_period(1.0)
# Start start_locally --> for easy of debugging & to avoid Agent overhead in Demo
optimizer.start_locally(job_complete_callback=job_complete_callback)
# optimizer.start(job_complete_callback=job_complete_callback)

# Wait for the Optimizer to finish
optimizer.wait()
top_exp = optimizer.get_top_experiments(top_k=1)[0]

task.get_logger().report_text(
    msg=f"ID of Top Experiment: {top_exp.id}"
)

top_model: Model = top_exp.models.output[-1]

# Add the Top Model as the Output model of the Optimizing Task
output_model: OutputModel = OutputModel()
output_model.update_design(top_model.config_text)
output_model.update_labels(top_model.labels)
output_model.update_weights(top_model.get_weights())
# Publish Task --> make it readonly/production ready
output_model.publish()
# Ensure that the Optimizer has stopped
optimizer.stop()
