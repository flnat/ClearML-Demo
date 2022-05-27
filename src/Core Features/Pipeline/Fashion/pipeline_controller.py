from clearml.automation import PipelineController


# Basic Pre-execution Callback for pipeline Task
def pre_exec_callback(pipeline: PipelineController, node: PipelineController.Node, param_override: dict) -> bool:
    print(
        "Cloning Task id={} with parameters: {}".format(
            node.base_task_id,
            param_override
        )
    )
    return True


# Basic Post-execution Callback for pipeline Task
def post_exec_callback(pipeline: PipelineController, node: PipelineController.Node) -> None:
    print(
        "Completed Task id={}".format(
            node.executed
        )
    )


# Instantiate Pipeline Controller
pipe: PipelineController = PipelineController(
    name="Fashion MNIST Pipeline",
    project="demo/Fashion MNIST",
    version="0.0.2",
    add_pipeline_tags=True,
)

# Showcase of manual parameter pass by
pipe.add_parameter(
    name="dataset_name",
    default="Fashion MNIST Matrices",
    param_type="str"
)

# Showcase of manual parameter pass by
pipe.add_parameter(
    name="dataset_project",
    default="demo/Fashion MNIST/datasets",
    param_type="str"
)

pipe.add_parameter(
    name="train_epochs",
    default="2",
    param_type="int"
)

pipe.add_parameter(
    name="layer_1_units",
    default="64",
    param_type="int"
)

pipe.add_parameter(
    name="layer_2_units",
    default="64",
    param_type="int"
)

pipe.set_default_execution_queue("default")
pipe.add_step(
    # Name of Task
    name="data_ingestion",
    # Name of template Task
    base_task_name="data_ingestion",
    # Location of template Task
    base_task_project="demo/Fashion MNIST",
    parameter_override={
        # Hyperparameter Injection through ${Task/pipeline.Object_Type.Name.Property}
        # in parameter_override

        "General/dataset_name": "${pipeline.dataset_name}",
        "General/dataset_project": "${pipeline.dataset_project}"
    },
    # Callbacks -->Logging/Reporting/Cleanup
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    # Step caching
    cache_executed_step=False
)

pipe.add_step(
    # Name of Task
    name="model_training",
    # Name of template Task
    base_task_name="model_training",
    # Dag Definition through parent-child hierarchy
    parents=["data_ingestion"],
    base_task_project="demo/Fashion MNIST",
    parameter_override={
        "General/train_data": "${data_ingestion.artifacts.train_data.url}",
        "General/train_labels": "${data_ingestion.artifacts.train_labels.url}",
        "General/epochs": "${pipeline.train_epochs}",
        "General/batch_size": 128,
        "General/layer_1_units": "${pipeline.layer_1_units}",
        "General/layer_2_units": "${pipeline.layer_2_units}"
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=False
)

pipe.add_step(
    # Name of Task
    name="initial_evaluation",
    # Name of template Task
    base_task_name="model_evaluation",
    base_task_project="demo/Fashion MNIST",
    parents=["model_training", "data_ingestion"],
    parameter_override={
        "General/model_id": "${model_training.models.output.-1.id}",
        "General/test_data": "${data_ingestion.artifacts.test_data.url}",
        "General/test_labels": "${data_ingestion.artifacts.test_labels.url}"
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=False
)

pipe.add_step(
    name="hyper_parameter_optimization",
    base_task_name="HyperParameter_Optimization",
    base_task_project="demo/Fashion MNIST",
    parents=["model_training", "initial_evaluation"],
    parameter_override={
        "General/template_task_id": "${model_training.id}",
        "General/max_no_jobs": 5,
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=False
)

# Tasks can be reused with new arguments
pipe.add_step(
    name="optimized_evaluation",
    base_task_name="model_evaluation",
    base_task_project="demo/Fashion MNIST",
    parents=["hyper_parameter_optimization", "data_ingestion"],
    parameter_override={
        "General/model_id": "${hyper_parameter_optimization.models.output.-1.id}",
        "General/test_data": "${data_ingestion.artifacts.test_data.url}",
        "General/test_labels": "${data_ingestion.artifacts.test_labels.url}"
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=False
)
# Run Pipline and Steps locally --> Debugging
pipe.start_locally(run_pipeline_steps_locally=True)
