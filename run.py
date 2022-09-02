"""Run a basic transformers text classificaiton workflow using AzureML pipelines.

TODO:
  - Use PipelineParameter to parameterize the workflow.
  - Use separate compute for train step, making sure train step works on GPU
"""
from pathlib import Path

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep


def find_or_create_compute_target(
    workspace,
    name="my-cluster",
    instance_type="Standard_DS11_v2",
    min_nodes=0,
    max_nodes=10,
    idle_seconds_before_scaledown=120,
    vm_priority="lowpriority",
):
    if name in workspace.compute_targets:
        return ComputeTarget(workspace=workspace, name=name)
    else:
        config = AmlCompute.provisioning_configuration(
            vm_size=instance_type,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            vm_priority=vm_priority,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown,
        )
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=True)
    return target


def main(
    workspace_config='config.json',
    experiment_name='basic-transformers-pipeline',
    prepare_script_path='prep/run.py',
    train_script_path='train/run.py',
):

    ws = Workspace.from_config(workspace_config)
    experiment = Experiment(ws, experiment_name)
    cpu_target = find_or_create_compute_target(
        ws,
        name='cpu-cluster',
        instance_type="Standard_DS11_v2",
        min_nodes=1,
        max_nodes=2,
        idle_seconds_before_scaledown=240,
        vm_priority="lowpriority",
    )

    datastore = ws.get_default_datastore()
    prepared_dataset = PipelineData("prepared_data", datastore=datastore).as_dataset()
    prepared_dataset = prepared_dataset.register(name="prepared_data")

    #########################
    # Prep Step
    #########################
    prep_script_path = Path(prepare_script_path)
    prep_env = Environment.from_conda_specification(
        "transformers_basic_prep_env", str(prep_script_path.parent / 'environment.yaml')
    )
    prepare_config = ScriptRunConfig(
        source_directory=prep_script_path.parent,
        script=prep_script_path.name,
        compute_target=cpu_target,
        environment=prep_env,
    )
    prepare_step = PythonScriptStep(
        name="Preparation Step",
        arguments=["--model_name_or_path", "distilbert-base-cased", "--seed", 42],
        outputs=[prepared_dataset],
        allow_reuse=True,
        runconfig=prepare_config.run_config,
        script_name=prep_script_path.name,
        source_directory=prep_script_path.parent,
    )

    #########################
    # Train Step
    #########################
    train_script_path = Path(train_script_path)
    train_env = Environment.from_conda_specification(
        "transformers_basic_train_env", str(train_script_path.parent / 'environment.yaml')
    )
    train_config = ScriptRunConfig(
        source_directory=train_script_path.parent,
        script=train_script_path.name,
        compute_target=cpu_target,
        environment=train_env,
    )
    train_step = PythonScriptStep(
        name="Training Step",
        arguments=["--model_name_or_path", "distilbert-base-cased", "--num_train_epochs", 2],
        inputs=[prepared_dataset.as_mount()],
        allow_reuse=False,
        runconfig=train_config.run_config,
        script_name=train_script_path.name,
        source_directory=train_script_path.parent,
    )

    pipeline = Pipeline(workspace=ws, steps=[prepare_step, train_step])

    experiment.submit(pipeline)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
