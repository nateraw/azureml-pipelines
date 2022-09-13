from pathlib import Path
from typing import Optional

import fire
from azureml.core import Experiment, Run
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.core.workspace import Workspace

from run import get_environment, get_workspace

def main(
    workspace_config: str = "config.json",
    experiment_name: str = "basic-transformers-pipeline",
    subscription_id: Optional[str] = None,
    resource_group: Optional[str] = None,
    workspace_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    environment_name="transformers_basic_train_env",
    requirements_file: str = "./basic_transformers_imdb/train/environment.yaml",
    run_id: str = None,
    model_artifact_path: str = "logs/saved_model",
    score_file: str = "./basic_transformers_imdb/score/run.py",
    model_name: str = "test-transformers-model",
    service_name: str = "test-transformers-service",
):
    # Init workspace object from azureml workspace resource you've created
    ws = get_workspace(
        workspace_config,
        subscription_id,
        resource_group,
        workspace_name,
        tenant_id,
        client_id,
        client_secret
    )

    # Point to an experiment
    experiment = Experiment(ws, name=experiment_name)

    run = Run(experiment, run_id)

    # Register your best run's model
    model = run.register_model(model_name=model_name, model_path=model_artifact_path)

    # Create an environment based on requirements
    env = get_environment(environment_name, requirements_file)

    # Get inference config to configure API's behavior
    inference_config = InferenceConfig(
        source_directory=Path(score_file).parent,
        entry_script=Path(score_file).name,
        environment=env,
        enable_gpu=False,
    )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1, memory_gb=1, description="A dummy transformers model deployment"
    )

    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
    )

    service.wait_for_deployment(True)


if __name__ == "__main__":
    fire.Fire(main)
