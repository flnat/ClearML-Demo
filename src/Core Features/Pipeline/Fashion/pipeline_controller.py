from clearml.automation import PipelineController


def pre_exec_callback(pipeline: PipelineController, node: PipelineController.Node, param_override: dict) -> bool:
    print(
        "Cloning Task id={} with parameters: {}".format(
            node.base_task_id,
            param_override
        )
    )
    return True


def post_exec_callback(pipeline: PipelineController, node: PipelineController.Node) -> None:
    print(
        "Completed Task id={}".format(
            node.executed
        )
    )


pipe: PipelineController = PipelineController(
    name="Fashion MNIST Pipeline",
    project="demo/Fashion MNIST",
    version="0.0.2",
    add_pipeline_tags=True,
)

pipe.add_parameter(
    name="dataset_name",
    default="Fashion MNIST Matrices",
    param_type="str"
)

pipe.add_parameter(
    name="dataset_project",
    default="demo/Fashion MNIST/datasets",
    param_type="str"
)

pipe.set_default_execution_queue("default")
pipe.add_step(
    name="data_ingestion",
    base_task_name="data_ingestion",
    base_task_project="demo/Fashion MNIST",
    parameter_override={
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/dataset_project": "${pipeline.dataset_project}"
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=True
)

pipe.add_step(
    name="model_training",
    base_task_name="model training",
    parents=["data_ingestion"],
    base_task_project="demo/Fashion MNIST",
    parameter_override={
        "General/train_data": "${data_ingestion.artifacts.train_data.url}",
        "General/train_labels": "${data_ingestion.artifacts.train_labels.url}",
        "General/epochs": 10,
        "General/batch_size": 128
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=True
)

pipe.add_step(
    name="model_evaluation",
    base_task_name="model_evaluation",
    parents=["model_training"],
    parameter_override={
        "General/model": "${model_training.models.output.-1.url}",
        "General/test_data": "${data_ingestion.artifacts.test_data.url}",
        "General/test_labels": "${data_ingestion.artifacts.test_labels.url}"
    },
    pre_execute_callback=pre_exec_callback,
    post_execute_callback=post_exec_callback,
    cache_executed_step=True
)

pipe.start_locally()
