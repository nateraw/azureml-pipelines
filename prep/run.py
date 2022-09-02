import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main(
    output_dir: str = os.environ.get("AZUREML_DATAREFERENCE_prepared_data", "./data_dir"),
    model_name_or_path: str = "distilbert-base-cased",
    seed: int = 42,
):
    ds = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    small_train_dataset = ds["train"].shuffle(seed=seed).select(range(1000)).map(tokenize_function, batched=True)
    small_eval_dataset = ds["test"].shuffle(seed=seed).select(range(1000)).map(tokenize_function, batched=True)

    small_train_dataset.save_to_disk(Path(output_dir) / "train")
    small_eval_dataset.save_to_disk(Path(output_dir) / "eval")


if __name__ == '__main__':
    import fire

    fire.Fire(main)
