import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main(
    output_dir: str = './data_dir',
    model_name_or_path: str = "distilbert-base-cased",
    seed: int = 42,
):
    print("\n\nOUTPUT DIR:", output_dir)

    ds = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    small_train_dataset = ds["train"].shuffle(seed=seed).select(range(100)).map(tokenize_function, batched=True)
    small_eval_dataset = ds["test"].shuffle(seed=seed).select(range(100)).map(tokenize_function, batched=True)

    output_dir = Path(output_dir)
    small_train_dataset.save_to_disk(output_dir / "train")
    small_eval_dataset.save_to_disk(output_dir / "eval")


if __name__ == '__main__':
    import fire

    fire.Fire(main)
