import os
from pathlib import Path

from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments)


def main(
    input_dir: str = os.environ.get('AZUREML_DATAREFERENCE_prepared_data', './data_dir'),
    output_dir: str = './outputs',
    logging_dir: str = './logs',
    num_train_epochs: int = 2,
    model_name_or_path: str = 'distilbert-base-cased',
):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    small_train_dataset = Dataset.load_from_disk(Path(input_dir) / 'train')
    small_eval_dataset = Dataset.load_from_disk(Path(input_dir) / 'eval')

    training_args = TrainingArguments(
        output_dir,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        report_to='tensorboard',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == '__main__':
    import fire

    fire.Fire(main)
