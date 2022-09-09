import json
import os
from pathlib import Path

import fire
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def init():
    global pipe

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "outputs/"))
    pipe = pipeline(model=str(model_dir), task='text-classification')


def run(input_data):
    input_data = json.loads(input_data)["data"]
    return pipe(input_data)


def main(endpoint: str = None):
    request_data = json.dumps({"data": "I really enjoyed this film!"})

    if endpoint is not None:
        response_data = requests.post(endpoint, request_data, headers={"Content-Type": "application/json"}).json()
    else:
        init()
        response_data = run(request_data)

    print(response_data)
    return response_data


if __name__ == "__main__":
    fire.Fire(main)