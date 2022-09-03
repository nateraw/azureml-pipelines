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
from azureml.core.runconfig import MpiConfiguration, DockerConfiguration
from azureml.core.authentication import ServicePrincipalAuthentication


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


def create_environment(environment_name: str, requirements_file: Path, docker_image=None):
    requirements_file = Path(requirements_file)
    if not requirements_file.exists():
        raise RuntimeError(f"Given requirements file '{requirements_file}' does not exist at the provided path")
    elif requirements_file.name.endswith(".txt"):
        env = Environment.from_pip_requirements(environment_name, requirements_file)
    elif requirements_file.name.endswith(".yml") or requirements_file.name.endswith(".yaml"):
        env = Environment.from_conda_specification(environment_name, requirements_file)
    else:
        print("Couldn't resolve env from requirements file")

    if docker_image:
        env.docker.base_image = docker_image

    return env


def get_workspace(
    workspace_config=None,
    subscription_id=None,
    resource_group=None,
    workspace_name=None,
    tenant_id=None,
    client_id=None,
    client_secret=None
):
    do_use_service_principal = all([tenant_id, client_id, client_secret])
    if do_use_service_principal:
        auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=client_id,
            service_principal_password=client_secret,
        )
    else:
        auth = None

    do_use_config_file = not all([subscription_id, resource_group, workspace_name])
    if do_use_config_file:
        ws = Workspace.from_config(workspace_config, auth=auth)
    else:
        ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            auth=auth
        )
    return ws


def create_step(
    script_path,
    name,
    compute_target,
    inputs=None,
    outputs=None,
    allow_reuse=False,
    arguments=None,
    num_nodes=1,
    environment_name=None,
    requirements_file=None,
    docker_image=None,
):
    script_path = Path(script_path)
    script_run_config = ScriptRunConfig(
        source_directory=script_path.parent,
        script=script_path.name,
        compute_target=compute_target,
        environment=create_environment(environment_name, requirements_file, docker_image),
        distributed_job_config=None if num_nodes == 1 else MpiConfiguration(process_count_per_node=1, node_count=num_nodes),
        docker_runtime_config=DockerConfiguration(use_docker=docker_image is not None),
    )
    step = PythonScriptStep(
        name=name,
        arguments=arguments,
        inputs=inputs,
        outputs=outputs,
        allow_reuse=allow_reuse,
        runconfig=script_run_config.run_config,
        script_name=script_path.name,
        source_directory=script_path.parent,
    )
    return step


def main(
    ################################
    # Prep Step Configuration
    ################################
    prepare_script_path='prep/run.py',
    prepare_environment_name='transformers_basic_prep_env',
    prepare_requirements_file='prep/environment.yaml',
    prepare_docker_image=None,
    prepare_compute_target_name='cpu-cluster',
    prepare_instance_type='Standard_DS11_v2',
    prepare_min_nodes=1,
    prepare_max_nodes=2,
    prepare_idle_seconds_before_scaledown=240,
    prepare_vm_priority='lowpriority',
    ################################
    # Train Step Configuration
    ################################
    train_script_path='train/run.py',
    train_environment_name='transformers_basic_train_env',
    train_requirements_file='train/environment.yaml',
    train_docker_image='mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04',
    train_compute_target_name='gpu-cluster',
    train_instance_type="Standard_NC6",
    train_min_nodes=1,
    train_max_nodes=1,
    train_idle_seconds_before_scaledown=240,
    train_vm_priority="dedicated",
    ################################
    # Workspace Configuration
    ################################
    workspace_config='config.json',
    experiment_name='basic-transformers-pipeline',
    subscription_id=None,
    resource_group=None,
    workspace_name=None,
    tenant_id=None,
    client_id=None,
    client_secret=None
):
    workspace = get_workspace(
        workspace_config,
        subscription_id,
        resource_group,
        workspace_name,
        tenant_id,
        client_id,
        client_secret
    )

    # Define pipeline data, the data that will be passed between steps
    datastore = workspace.get_default_datastore()
    prepared_dataset = PipelineData("prepared_data", datastore=datastore).as_dataset()
    prepared_dataset = prepared_dataset.register(name="prepared_data")

    prep_step = create_step(
        prepare_script_path,
        name="Preparation Step",
        compute_target=find_or_create_compute_target(
            workspace,
            name=prepare_compute_target_name,
            instance_type=prepare_instance_type,
            min_nodes=prepare_min_nodes,
            max_nodes=prepare_max_nodes,
            idle_seconds_before_scaledown=prepare_idle_seconds_before_scaledown,
            vm_priority=prepare_vm_priority,
        ),
        inputs=None,
        outputs=[prepared_dataset],
        allow_reuse=True,
        arguments=["--model_name_or_path", "distilbert-base-cased", "--seed", 42],
        environment_name=prepare_environment_name,
        requirements_file=prepare_requirements_file,
        docker_image=prepare_docker_image,
    )
    train_step = create_step(
        train_script_path,
        name="Training Step",
        compute_target=find_or_create_compute_target(
            workspace,
            name=train_compute_target_name,
            instance_type=train_instance_type,
            min_nodes=train_min_nodes,
            max_nodes=train_max_nodes,
            idle_seconds_before_scaledown=train_idle_seconds_before_scaledown,
            vm_priority=train_vm_priority,
        ),
        outputs=None,
        allow_reuse=False,
        arguments=["--input_dir", prepared_dataset.as_mount(), "--model_name_or_path", "distilbert-base-cased", "--num_train_epochs", 2, "--seed", 1234],
        environment_name=train_environment_name,
        requirements_file=train_requirements_file,
        docker_image=train_docker_image,
    )
    pipeline = Pipeline(workspace=workspace, steps=[prep_step, train_step])

    experiment = Experiment(workspace, experiment_name)
    experiment.submit(pipeline)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
